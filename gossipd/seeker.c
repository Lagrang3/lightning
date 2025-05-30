/* This contains the code which actively seeks out gossip from peers */
#include "config.h"
#include <bitcoin/chainparams.h>
#include <ccan/array_size/array_size.h>
#include <ccan/asort/asort.h>
#include <ccan/intmap/intmap.h>
#include <ccan/tal/str/str.h>
#include <common/daemon_conn.h>
#include <common/decode_array.h>
#include <common/gossmap.h>
#include <common/memleak.h>
#include <common/pseudorand.h>
#include <common/random_select.h>
#include <common/status.h>
#include <common/timeout.h>
#include <gossipd/gossipd.h>
#include <gossipd/gossipd_wiregen.h>
#include <gossipd/gossmap_manage.h>
#include <gossipd/queries.h>
#include <gossipd/seeker.h>

#define GOSSIP_SEEKER_INTERVAL(seeker) \
	DEV_FAST_GOSSIP((seeker)->daemon->dev_fast_gossip, 5, 60)

#define GOSSIP_SEEKER_RESYNC_INTERVAL(seeker) \
	DEV_FAST_GOSSIP((seeker)->daemon->dev_fast_gossip, 30, 3600)

#define SEEKER_GOSSIPERS 10

enum seeker_state {
	/* Still streaming gossip from single peer. */
	STARTING_UP,

	/* Probing: checking our startup really is finished. */
	PROBING_SCIDS,

	/* Probing: check that we have node_announcements. */
	PROBING_NANNOUNCES,

	/* Normal running. */
	NORMAL,

	/* Asking a peer for unknown scids. */
	ASKING_FOR_UNKNOWN_SCIDS,

	/* Asking a peer for stale scids. */
	ASKING_FOR_STALE_SCIDS,
};

/* Gossip we're seeking at the moment. */
struct seeker {
	struct daemon *daemon;

	enum seeker_state state;

	/* Timer which checks on progress every minute */
	struct oneshot *check_timer;

	/* Full sync gossip from one peer every hour */
	struct oneshot *sync_timer;

	/* Channels we've heard about, but don't know (by scid). */
	UINTMAP(bool) unknown_scids;

	/* Channels we've heard about newer timestamps for (by scid).  u8 is
	 * query_flags. */
	UINTMAP(u8 *) stale_scids;

	/* Range of scid blocks we've probed. */
	size_t scid_probe_start, scid_probe_end;

	/* During startup, we ask a single peer for gossip (set to NULL if peer dies)*/
	struct peer *random_peer;

	/* The last peer we requested a full gossip sync from. */
	struct peer *last_full_sync_peer;

	/* This checks progress of our random peer */
	size_t prev_gossip_count;

	/* Array of scids for node announcements. */
	struct short_channel_id *nannounce_scids;
	u8 *nannounce_query_flags;

	/* Are there any node_ids we didn't know?  Implies we're
	 * missing channels. */
	bool unknown_nodes;

	/* Peers we've asked to stream us gossip (set to NULL if peer dies) */
	struct peer **gossiper;

	/* A peer that told us about unknown gossip (set to NULL if peer dies). */
	struct peer *preferred_peer;

	/* check_timer cycles since streaming began from the last new gossiper. */
	u8 new_gossiper_elapsed;
};

/* Mutual recursion */
static void seeker_check(struct seeker *seeker);
static void probe_some_random_scids(struct seeker *seeker);

static void begin_check_timer(struct seeker *seeker)
{
	const u32 polltime = GOSSIP_SEEKER_INTERVAL(seeker);

	seeker->check_timer = new_reltimer(&seeker->daemon->timers,
					   seeker,
					   time_from_sec(polltime),
					   seeker_check, seeker);
}

/* Set this peer as our random peer; return false if NULL. */
static bool selected_peer(struct seeker *seeker, struct peer *peer)
{
	if (!peer)
		return false;

	seeker->random_peer = peer;

	/* Give it some grace in case we immediately hit timer */
	seeker->prev_gossip_count
		= peer->query_reply_counter - GOSSIP_SEEKER_INTERVAL(seeker);
	return true;
}

#define set_state(seeker, state, peer, ...)				\
	set_state_((seeker), (state), (peer), stringify(state), __VA_ARGS__)

static void set_state_(struct seeker *seeker, enum seeker_state state,
		       struct peer *peer,
		       const char *statename, const char *fmt, ...)
PRINTF_FMT(5,6);

static void set_state_(struct seeker *seeker, enum seeker_state state,
		       struct peer *peer,
		       const char *statename, const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	status_peer_debug(peer ? &peer->id : NULL,
			  "seeker: state = %s %s",
			  statename, tal_vfmt(tmpctx, fmt, ap));
	va_end(ap);
	seeker->state = state;
	selected_peer(seeker, peer);
}

struct seeker *new_seeker(struct daemon *daemon)
{
	struct seeker *seeker = tal(daemon, struct seeker);

	seeker->daemon = daemon;
	uintmap_init(&seeker->unknown_scids);
	uintmap_init(&seeker->stale_scids);
	seeker->random_peer = NULL;
	u32 gossipers = daemon->autoconnect_seeker_peers > SEEKER_GOSSIPERS ?
			daemon->autoconnect_seeker_peers : SEEKER_GOSSIPERS;
	seeker->gossiper = tal_arrz(seeker, struct peer *, gossipers);
	seeker->preferred_peer = NULL;
	seeker->unknown_nodes = false;
	seeker->last_full_sync_peer = NULL;
	seeker->new_gossiper_elapsed = 0;
	set_state(seeker, STARTING_UP, NULL, "New seeker");
	begin_check_timer(seeker);
	seeker->sync_timer = NULL;
	return seeker;
}

static void set_preferred_peer(struct seeker *seeker, struct peer *peer)
{
	seeker->preferred_peer = peer;
}

/* Get a random peer, but try our preferred peer first, if any.  This
 * biasses us to the peer that told us of unexpected gossip. */
static struct peer *random_seeker(struct seeker *seeker,
				  bool (*check_peer)(const struct peer *peer))
{
	struct peer *peer = seeker->preferred_peer;
	struct peer *first;
	struct peer_node_id_map_iter it;

	/* 80% chance of immediately choosing a peer who reported the missing
	 * stuff: they presumably can tell us more about it.  We don't
	 * *always* choose it because it could be simply spamming us with
	 * invalid announcements to get chosen, and we don't handle that case
	 * well yet. */
	if (peer && check_peer(peer) && pseudorand(5) != 0) {
		seeker->random_peer = NULL;
		return peer;
	}

	/* Rotate through, so we don't favor a single peer. */
	peer = first = first_random_peer(seeker->daemon, &it);
	while (peer) {
		if (check_peer(peer))
			break;
		peer = next_random_peer(seeker->daemon, first, &it);
	}
	return peer;
}

static bool peer_made_progress(struct seeker *seeker, const struct peer *peer)
{
	/* Has it made progress (at least one valid update per second)?  If
	 * not, we assume it's finished, and if it hasn't, we'll end up
	 * querying backwards in next steps. */
	if (peer->query_reply_counter
	    >= seeker->prev_gossip_count + GOSSIP_SEEKER_INTERVAL(seeker)) {
		seeker->prev_gossip_count = peer->query_reply_counter;
		return true;
	}

	return false;
}

static void disable_gossip_stream(struct seeker *seeker, struct peer *peer)
{
	u8 *msg;

	status_peer_debug(&peer->id, "seeker: disabling gossip");

	/* This is allowed even if they don't understand it (odd) */
	msg = towire_gossip_timestamp_filter(NULL,
					     &chainparams->genesis_blockhash,
					     UINT32_MAX,
					     UINT32_MAX);
	queue_peer_msg(peer->daemon, &peer->id, take(msg));
}

static void enable_gossip_stream(struct seeker *seeker, struct peer *peer,
				 bool ask_for_all)
{
	u32 start;
	u8 *msg;

	/* If we have no gossip, always ask for everything */
	if (!gossmap_manage_populated(peer->daemon->gm))
		ask_for_all = true;

	/* Modern timestamp_filter is a trinary: 0 = all, FFFFFFFF = none,
	 * other = from now on */
	if (ask_for_all) {
		start = 0;
	} else {
		/* Just in case they care */
		start = time_now().ts.tv_sec - GOSSIP_SEEKER_INTERVAL(seeker) * 10;
	}

	status_peer_debug(&peer->id, "seeker: starting gossip (%s)",
			  ask_for_all ? "EVERYTHING" : "streaming");

	/* This is allowed even if they don't understand it (odd) */
	msg = towire_gossip_timestamp_filter(NULL,
					     &chainparams->genesis_blockhash,
					     start,
					     UINT32_MAX);
	queue_peer_msg(peer->daemon, &peer->id, take(msg));
}

static void normal_gossip_start(struct seeker *seeker, struct peer *peer, bool ask_for_all)
{
	bool enable_stream = false;

	/* BOLT-remove-old-features #7:
	 * Understanding of messages used to be indicated with the `gossip_queries`
	 * feature bit; now these messages are universally supported, that feature has
	 * now been slightly repurposed.  Not offering this feature means a node is not
	 * worth querying for gossip: either they do not store the entire gossip map, or
	 * they are only connected to a single peer (this one).
	 */
	if (!peer->gossip_queries_feature)
		return;

	/* Make this one of our streaming gossipers if we aren't full */
	for (size_t i = 0; i < tal_count(seeker->gossiper); i++) {
		if (seeker->gossiper[i] == NULL) {
			seeker->gossiper[i] = peer;
			enable_stream = true;
			break;
		}
	}

	if (enable_stream)
		enable_gossip_stream(seeker, peer, ask_for_all);
	else
		disable_gossip_stream(seeker, peer);
}

/* Turn unknown_scids map into a flat array, removes from map. */
static struct short_channel_id *unknown_scids_remove(const tal_t *ctx,
						     struct seeker *seeker)
{
	struct short_channel_id *scids;
	/* Marshal into an array: we can fit 8000 comfortably. */
	size_t i, max = 8000;
	u64 scid;

	scids = tal_arr(ctx, struct short_channel_id, max);
	i = 0;
	while (uintmap_first(&seeker->unknown_scids, &scid)) {
		scids[i].u64 = scid;
		(void)uintmap_del(&seeker->unknown_scids, scid);
		if (++i == max)
			break;
	}
	tal_resize(&scids, i);
	return scids;
}

/* We have selected this peer to stream us startup gossip */
static void peer_gossip_startup(struct seeker *seeker, struct peer *peer,
				bool ask_for_all)
{
	status_peer_debug(&peer->id, "seeker: chosen as startup peer");
	selected_peer(seeker, peer);
	normal_gossip_start(seeker, peer, ask_for_all);
}

static bool peer_has_gossip_queries(const struct peer *peer)
{
	return peer->gossip_queries_feature;
}

static bool peer_can_take_range_query(const struct peer *peer)
{
	return peer->gossip_queries_feature
		&& !peer->range_replies;
}

static bool peer_can_take_scid_query(const struct peer *peer)
{
	return peer->gossip_queries_feature
		&& !peer->scid_query_outstanding;
}

static void scid_query_done(struct peer *peer, bool complete)
{
	struct seeker *seeker = peer->daemon->seeker;

	/* Peer completed!  OK, start random scid probe in case we're
	 * still missing gossip. */
	probe_some_random_scids(seeker);
}

/* Returns true if there were scids to seek. */
static bool seek_any_unknown_scids(struct seeker *seeker)
{
	struct peer *peer;
	struct short_channel_id *scids;

	/* Nothing we need to know about? */
	if (uintmap_empty(&seeker->unknown_scids))
		return false;

	/* No peers can answer?  Try again later. */
	peer = random_seeker(seeker, peer_can_take_scid_query);
	if (!peer)
		return false;

	scids = unknown_scids_remove(tmpctx, seeker);
	set_state(seeker, ASKING_FOR_UNKNOWN_SCIDS, peer,
		  "Asking for %zu scids", tal_count(scids));
	if (!query_short_channel_ids(seeker->daemon, peer, scids, NULL,
				     scid_query_done))
		status_failed(STATUS_FAIL_INTERNAL_ERROR,
			      "seeker: quering %zu scids is too many?",
			      tal_count(scids));
	return true;
}

/* Turns stale_scid_map into two arrays, and removes from map */
static struct short_channel_id *stale_scids_remove(const tal_t *ctx,
						   struct seeker *seeker,
						   u8 **query_flags)
{
	struct short_channel_id *scids;
	const u8 *qf;
	/* We can fit 7000 comfortably (8 byte scid, 1 byte flag). */
	size_t i, max = 7000;
	u64 scid;

	scids = tal_arr(ctx, struct short_channel_id, max);
	*query_flags = tal_arr(ctx, u8, max);

	i = 0;
	while ((qf = uintmap_first(&seeker->stale_scids, &scid)) != NULL) {
		scids[i].u64 = scid;
		(*query_flags)[i] = *qf;
		uintmap_del(&seeker->stale_scids, scid);
		tal_free(qf);
		i++;
		if (i == max)
			break;
	}
	tal_resize(&scids, i);
	tal_resize(query_flags, i);
	return scids;
}

static bool seek_any_stale_scids(struct seeker *seeker)
{
	struct peer *peer;
	struct short_channel_id *scids;
	u8 *query_flags;

	/* Nothing we need to know about? */
	if (uintmap_empty(&seeker->stale_scids))
		return false;

	/* No peers can answer?  Try again later. */
	peer = random_seeker(seeker, peer_can_take_scid_query);
	if (!peer)
		return false;

	/* This is best-effort, so this consumes them as well. */
	scids = stale_scids_remove(tmpctx, seeker, &query_flags);
	set_state(seeker, ASKING_FOR_STALE_SCIDS, peer,
		  "Asking for %zu scids", tal_count(scids));

	if (!query_short_channel_ids(seeker->daemon, peer, scids, query_flags,
				     scid_query_done))
		status_failed(STATUS_FAIL_INTERNAL_ERROR,
			      "seeker: quering %zu scids is too many?",
			      tal_count(scids));
	return true;
}

/* Returns true and sets first_blocknum and number_of_blocks if
 * there's more to find. */
static bool next_block_range(struct seeker *seeker,
			     u32 prev_num_blocks,
			     u32 *first_blocknum, u32 *number_of_blocks)
{
	const u32 current_height = seeker->daemon->current_blockheight;

	/* We always try to get twice as many as last time. */
	*number_of_blocks = prev_num_blocks * 2;

	if (seeker->scid_probe_start > 0) {
		/* Enlarge probe to cover prior blocks, but twice as many. */
		if (*number_of_blocks > seeker->scid_probe_start) {
			*number_of_blocks = seeker->scid_probe_start;
			*first_blocknum = 0;
		} else {
			*first_blocknum
				= seeker->scid_probe_start - *number_of_blocks;
		}
		seeker->scid_probe_start = *first_blocknum;
		return true;
	}

	/* We allow 6 new blocks since we started; they should be empty anyway */
	if (seeker->scid_probe_end + 6 < current_height) {
		if (seeker->scid_probe_end + *number_of_blocks > current_height)
			*number_of_blocks
				= current_height - seeker->scid_probe_end;
		*first_blocknum = seeker->scid_probe_end + 1;
		seeker->scid_probe_end = *first_blocknum + *number_of_blocks - 1;
		return true;
	}

	/* No more to find. */
	return false;
}

static int cmp_scid(const struct short_channel_id *a,
		    const struct short_channel_id *b,
		    void *unused)
{
	if (a->u64 > b->u64)
		return 1;
	else if (a->u64 < b->u64)
		return -1;
	return 0;
}

/* We can't ask for channels by node_id, so probe at random */
static bool get_unannounced_nodes(const tal_t *ctx,
				  struct daemon *daemon,
				  size_t max,
				  struct short_channel_id **scids,
				  u8 **query_flags)
{
	size_t num = 0, off, max_idx;
	struct gossmap *gossmap = gossmap_manage_get_gossmap(daemon->gm);

	/* No nodes?  Nothing to ask for */
	max_idx = gossmap_max_node_idx(gossmap);
	if (max_idx == 0 || max == 0)
		return false;

	/* Start at random index */
	off = pseudorand(max_idx);
	*scids = tal_arr(ctx, struct short_channel_id, max);

	for (size_t i = 0; i < max_idx; i++) {
		const struct gossmap_node *node = gossmap_node_byidx(gossmap, (off + i) % max_idx);
		const struct gossmap_chan *chan;

		if (!node)
			continue;
		if (gossmap_node_announced(node))
			continue;

		/* Query first chan. */
		chan = gossmap_nth_chan(gossmap, node, 0, NULL);
		(*scids)[num++] = gossmap_chan_scid(gossmap, chan);

		if (num == max)
			break;
	}

	if (num == 0) {
		*scids = tal_free(*scids);
		return false;
	}

	if (num < max)
		tal_resize(scids, num);

	/* Sort them into order, and remove duplicates! */
	asort(*scids, num, cmp_scid, NULL);
	for (size_t i = 1; i < tal_count(*scids); i++) {
		if (short_channel_id_eq((*scids)[i], (*scids)[i-1])) {
			tal_arr_remove(scids, i);
		}
	}

	/* Now get flags. */
	*query_flags = tal_arr(ctx, u8, tal_count(*scids));
	for (size_t i = 0; i < tal_count(*scids); i++) {
		const struct gossmap_chan *chan = gossmap_find_chan(gossmap, &(*scids)[i]);

		(*query_flags)[i] = 0;
		if (!gossmap_node_announced(gossmap_nth_node(gossmap, chan, 0)))
			(*query_flags)[i] |= SCID_QF_NODE1;
		if (!gossmap_node_announced(gossmap_nth_node(gossmap, chan, 1)))
			(*query_flags)[i] |= SCID_QF_NODE2;
	}
	return true;
}

/* Mutual recursion */
static void peer_gossip_probe_nannounces(struct seeker *seeker);

static void nodeannounce_query_done(struct peer *peer, bool complete)
{
	struct daemon *daemon = peer->daemon;
	struct seeker *seeker = daemon->seeker;
	size_t new_nannounce = 0, num_scids;
	struct gossmap *gossmap = gossmap_manage_get_gossmap(daemon->gm);

	/* We might have given up on them, then they replied. */
	if (seeker->random_peer != peer) {
		status_peer_debug(&peer->id, "seeker: belated reply: ignoring");
		return;
	}

	seeker->random_peer = NULL;

	num_scids = tal_count(seeker->nannounce_scids);
	for (size_t i = 0; i < num_scids; i++) {
		struct gossmap_chan *c = gossmap_find_chan(gossmap,
							   &seeker->nannounce_scids[i]);
		/* Could have closed since we asked. */
		if (!c)
			continue;
		if ((seeker->nannounce_query_flags[i] & SCID_QF_NODE1)
		    && gossmap_node_announced(gossmap_nth_node(gossmap, c, 0)))
			new_nannounce++;
		if ((seeker->nannounce_query_flags[i] & SCID_QF_NODE2)
		    && gossmap_node_announced(gossmap_nth_node(gossmap, c, 1)))
			new_nannounce++;
	}

	status_peer_debug(&peer->id,
			  "seeker: found %zu new node_announcements in %zu scids",
			  new_nannounce, num_scids);

	seeker->nannounce_scids = tal_free(seeker->nannounce_scids);
	seeker->nannounce_query_flags = tal_free(seeker->nannounce_query_flags);

	if (!new_nannounce) {
		set_state(seeker, NORMAL, NULL,
			  "No new node_announcements in %zu scids", num_scids);
		return;
	}

	/* Since they told us about new announcements, keep asking them. */
	set_preferred_peer(seeker, peer);

	/* Double every time.  We may skip a few, of course, since map
	 * is changing. */
	num_scids *= 2;
	/* Don't try to create a query larger than 64k */
	if (num_scids > 7000)
		num_scids = 7000;

	if (!get_unannounced_nodes(seeker, seeker->daemon, num_scids,
				   &seeker->nannounce_scids,
				   &seeker->nannounce_query_flags)) {
		/* Nothing unknown at all?  Great, we're done */
		set_state(seeker, NORMAL, NULL, "No unannounced nodes");
		return;
	}

	peer_gossip_probe_nannounces(seeker);
}

/* Pick a peer, ask it for a few node announcements, to check. */
static void peer_gossip_probe_nannounces(struct seeker *seeker)
{
	struct peer *peer;

	peer = random_seeker(seeker, peer_can_take_scid_query);
	set_state(seeker, PROBING_NANNOUNCES, peer,
		  "Probing for %zu scids",
		  tal_count(seeker->nannounce_scids));
	if (!peer)
		return;

	if (!query_short_channel_ids(seeker->daemon, peer,
				     seeker->nannounce_scids,
				     seeker->nannounce_query_flags,
				     nodeannounce_query_done))
		status_failed(STATUS_FAIL_INTERNAL_ERROR,
			      "seeker: quering %zu scids is too many?",
			      tal_count(seeker->nannounce_scids));
}

/* They have update with this timestamp: do we want it? */
static bool want_update(struct gossmap *gossmap,
			u32 timestamp,
			const struct gossmap_chan *chan,
			int dir)
{
	u32 our_timestamp;

	if (!gossmap_chan_set(chan, dir))
		return timestamp != 0;

	gossmap_chan_get_update_details(gossmap, chan, dir, &our_timestamp,
					NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	if (timestamp <= our_timestamp)
		return false;

	return true;
}

/* They gave us timestamps.  Do we want updated versions? */
static void check_timestamps(struct seeker *seeker,
			     struct gossmap *gossmap,
			     struct gossmap_chan *c,
			     const struct channel_update_timestamps *ts,
			     struct peer *peer)
{
	u8 *stale;
	u8 query_flag = 0;
	struct short_channel_id scid;

	/* BOLT #7:
	 * * `timestamp_node_id_1` is the timestamp of the `channel_update`
	 *    for `node_id_1`, or 0 if there was no `channel_update` from that
	 *    node.
	 * * `timestamp_node_id_2` is the timestamp of the `channel_update`
	 *    for `node_id_2`, or 0 if there was no `channel_update` from that
	 *    node.
	 */
	if (want_update(gossmap, ts->timestamp_node_id_1, c, 0))
		query_flag |= SCID_QF_UPDATE1;
	if (want_update(gossmap, ts->timestamp_node_id_2, c, 1))
		query_flag |= SCID_QF_UPDATE2;

	if (!query_flag)
		return;

	/* Add in flags if we're already getting it. */
	scid = gossmap_chan_scid(gossmap, c);
	stale = uintmap_get(&seeker->stale_scids, scid.u64);
	if (!stale) {
		stale = talz(seeker, u8);
		uintmap_add(&seeker->stale_scids, scid.u64, stale);
		set_preferred_peer(seeker, peer);
	}
	*stale |= query_flag;
}

static bool add_unknown_scid(struct seeker *seeker,
			     struct short_channel_id scid,
			     struct peer *peer)
{
	/* Check we're not already getting this one. */
	if (!uintmap_add(&seeker->unknown_scids, scid.u64, true))
		return false;

	set_preferred_peer(seeker, peer);
	return true;
}

static void process_scid_probe(struct peer *peer,
			       u32 first_blocknum, u32 number_of_blocks,
			       const struct range_query_reply *replies)
{
	struct daemon *daemon = peer->daemon;
	struct seeker *seeker = daemon->seeker;
	bool new_unknown_scids = false;
	struct gossmap *gossmap;

	/* We might have given up on them, then they replied. */
	if (seeker->random_peer != peer)
		return;

	seeker->random_peer = NULL;
	gossmap = gossmap_manage_get_gossmap(daemon->gm);

	for (size_t i = 0; i < tal_count(replies); i++) {
		struct gossmap_chan *c = gossmap_find_chan(gossmap,
							   &replies[i].scid);
		if (c) {
			check_timestamps(seeker, gossmap, c, &replies[i].ts, peer);
			continue;
		}

		new_unknown_scids |= add_unknown_scid(seeker,
						      replies[i].scid,
						      peer);
	}

	/* No new unknown scids, or no more to ask?  We give some wiggle
	 * room in case blocks came in since we started. */
	if (new_unknown_scids
	    && next_block_range(seeker, number_of_blocks,
				&first_blocknum, &number_of_blocks)) {
		/* This must return a peer, since we have the current peer! */
		peer = random_seeker(seeker, peer_can_take_range_query);
		assert(peer);
		selected_peer(seeker, peer);

		query_channel_range(seeker->daemon, peer,
				    first_blocknum, number_of_blocks,
				    QUERY_ADD_TIMESTAMPS,
				    process_scid_probe);
		return;
	}

	/* Channel probe finished, try asking for 128 unannounced nodes. */
	if (!get_unannounced_nodes(seeker, seeker->daemon, 128,
				   &seeker->nannounce_scids,
				   &seeker->nannounce_query_flags)) {
		/* No unknown nodes.  Great! */
		set_state(seeker, NORMAL, NULL, "No unannounced nodes");
		return;
	}

	peer_gossip_probe_nannounces(seeker);
}

/* Pick a peer, ask it for a few scids, to check. */
static void peer_gossip_probe_scids(struct seeker *seeker)
{
	struct peer *peer;

	peer = random_seeker(seeker, peer_can_take_range_query);
	set_state(seeker, PROBING_SCIDS, peer,
		  "Seeking scids %zu - %zu",
		  seeker->scid_probe_start, seeker->scid_probe_end);
	if (!peer)
		return;

	/* This calls process_scid_probe when we get the reply. */
	query_channel_range(seeker->daemon, peer,
			    seeker->scid_probe_start,
			    seeker->scid_probe_end - seeker->scid_probe_start + 1,
			    QUERY_ADD_TIMESTAMPS,
			    process_scid_probe);
}

static void probe_random_scids(struct seeker *seeker, size_t num_blocks)
{
	u32 avail_blocks;

	/* Ignore early blocks (unless we're before, which would be weird) */
	if (seeker->daemon->current_blockheight
	    < chainparams->when_lightning_became_cool)
		avail_blocks = seeker->daemon->current_blockheight;
	else
		avail_blocks = seeker->daemon->current_blockheight
			- chainparams->when_lightning_became_cool;

	if (avail_blocks < num_blocks) {
		seeker->scid_probe_start = 0;
		seeker->scid_probe_end = seeker->daemon->current_blockheight;
	} else {
		seeker->scid_probe_start
			= chainparams->when_lightning_became_cool
			+ pseudorand(avail_blocks - num_blocks);
		seeker->scid_probe_end
			= seeker->scid_probe_start + num_blocks - 1;
	}

	seeker->nannounce_scids = NULL;
	peer_gossip_probe_scids(seeker);
}

/* We usually get a channel per block, so these cover a fair bit of ground */
static void probe_some_random_scids(struct seeker *seeker)
{
	return probe_random_scids(seeker, 1024);
}

static void probe_many_random_scids(struct seeker *seeker)
{
	return probe_random_scids(seeker, 10000);
}

/* Find a random node with an announcement. */
static struct node_id *get_random_node(const tal_t *ctx,
				       struct seeker *seeker)
{
	struct gossmap *gossmap = gossmap_manage_get_gossmap(seeker->daemon->gm);
	struct gossmap_node *node = gossmap_random_node(gossmap);

	if (!node)
		return NULL;

	for (int i = 0; i<20; i++) {
		struct node_id id;

		gossmap_node_get_id(gossmap, node, &id);

		/* Make sure it *has* an announcement, and we're not
		 * already connected */
		if (!node_id_eq(&id, &seeker->daemon->id)
		    && gossmap_node_get_announce(tmpctx, gossmap, node)
		    && !find_peer(seeker->daemon, &id)) {
			return tal_dup(ctx, struct node_id, &id);
		}

		node = gossmap_next_node(gossmap, node);
		if (!node)
			node = gossmap_first_node(gossmap);
	}

	return NULL;
}

/* Ask lightningd for more peers if we're short on gossip streamers. */
static void maybe_get_new_peer(struct seeker *seeker)
{
	size_t connected_peers = peer_node_id_map_count(seeker->daemon->peers);
	struct node_id *random_node;

	/* Respect user-defined autoconnect peer limit. */
	if (connected_peers >= seeker->daemon->autoconnect_seeker_peers)
		return;

	status_debug("seeker: need more peers for gossip (have %zu)",
		     connected_peers);

	random_node = get_random_node(tmpctx, seeker);
	if (!random_node) {
		status_debug("seeker: no more potential peers found");
		return;
	}

	u8 *msg = towire_gossipd_connect_to_peer(NULL, random_node);
	daemon_conn_send(seeker->daemon->master, take(msg));
	tal_free(random_node);
}

static void check_firstpeer(struct seeker *seeker)
{
	struct peer *peer = seeker->random_peer;
	struct peer_node_id_map_iter it;

	/* It might have died, pick another. */
	if (!peer) {
		peer = random_seeker(seeker, peer_has_gossip_queries);
		/* No peer?  Wait for a new one to join. */
		if (!peer) {
			status_debug("seeker: no peers, waiting");
			if (seeker->new_gossiper_elapsed)
				maybe_get_new_peer(seeker);
			else
				seeker->new_gossiper_elapsed++;
			return;
		}

		peer_gossip_startup(seeker, peer, false);
		return;
	}

	/* If no progress, we assume it's finished, and if it hasn't,
	 * we'll end up querying backwards in next steps. */
	if (peer_made_progress(seeker, peer))
		return;

	/* Other peers can gossip now. */
	status_peer_debug(&peer->id, "seeker: startup peer finished");
	seeker->random_peer = NULL;

	for (struct peer *p = peer_node_id_map_first(seeker->daemon->peers, &it);
	     p;
	     p = peer_node_id_map_next(seeker->daemon->peers, &it)) {
		if (p == peer)
			continue;

		normal_gossip_start(seeker, p, false);
	}

	/* Ask a random peer for all channels, in case we're missing */
	seeker->scid_probe_start = chainparams->when_lightning_became_cool;
	seeker->scid_probe_end = seeker->daemon->current_blockheight;
	if (seeker->scid_probe_start > seeker->scid_probe_end)
		seeker->scid_probe_start = 0;
	peer_gossip_probe_scids(seeker);
}

static void check_probe(struct seeker *seeker,
			void (*restart)(struct seeker *seeker))
{
	struct peer *peer = seeker->random_peer;

	/* It might have died, pick another. */
	if (!peer) {
		restart(seeker);
		return;
	}

	/* Is peer making progress with responses? */
	if (peer_made_progress(seeker, peer))
		return;

	status_peer_debug(&peer->id,
			  "has only moved gossip %zu->%zu for probe, giving up on it",
			  seeker->prev_gossip_count, peer->query_reply_counter);
	seeker->random_peer = NULL;
	restart(seeker);
}

static bool peer_is_not_gossipper(const struct peer *peer)
{
	const struct seeker *seeker = peer->daemon->seeker;

	/* BOLT-remove-old-features #7:
	 * `gossip_queries`... Not offering this feature means a node is not
	 * worth querying for gossip
	 */
	if (!peer->gossip_queries_feature)
		return false;

	for (size_t i = 0; i < tal_count(seeker->gossiper); i++) {
		if (seeker->gossiper[i] == peer)
			return false;
	}
	return true;
}

/* Allows evaluation of least useful gossip streamer. */
static void reset_gossip_performance_metrics(struct seeker *seeker)
{
	seeker->new_gossiper_elapsed = 0;
	for (size_t i = 0; i < tal_count(seeker->gossiper); i++) {
		if (seeker->gossiper[i])
			seeker->gossiper[i]->gossip_counter = 0;
	}
}

static void maybe_rotate_gossipers(struct seeker *seeker)
{
	struct peer *peer;
	size_t i, lowest_idx;

	seeker->new_gossiper_elapsed++;

	/* If all (usable) peers are gossiping, we're done */
	peer = random_seeker(seeker, peer_is_not_gossipper);
	if (!peer)
		return;

	/* If we have a slot free, fill it. */
	for (i = 0; i < tal_count(seeker->gossiper); i++) {
		if (!seeker->gossiper[i]) {
			status_peer_debug(&peer->id, "seeker: filling slot %zu",
					  i);
			lowest_idx = i;
			goto set_gossiper;
		}
	}
	/* Otherwise, rotate out worst gossiper every 30 minutes on average. */
	if (pseudorand(25) != 0)
		return;
	/* Don't evaluate gossip performance at a faster rate than
	 * new gossip is periodically emitted. */
	if (seeker->new_gossiper_elapsed < 5)
		return;
	u32 lowest_count = UINT_MAX;
	lowest_idx = 0;
	for (size_t j = 0; j < tal_count(seeker->gossiper); j++) {
		if (seeker->gossiper[j]->gossip_counter < lowest_count) {
			lowest_count = seeker->gossiper[j]->gossip_counter;
			lowest_idx = j;
		}
	}
	status_debug("seeker: ejecting worst gossiper %s - slot %zu: "
		     "novel gossip count %zu over %u minutes",
		     fmt_node_id(tmpctx, &seeker->gossiper[lowest_idx]->id),
		     lowest_idx, seeker->gossiper[lowest_idx]->gossip_counter,
		     seeker->new_gossiper_elapsed);
	status_peer_debug(&peer->id, "seeker: replacing slot %zu",
			  lowest_idx);
	disable_gossip_stream(seeker, seeker->gossiper[lowest_idx]);

set_gossiper:
	seeker->gossiper[lowest_idx] = peer;
	enable_gossip_stream(seeker, peer, false);
	reset_gossip_performance_metrics(seeker);
}

static bool seek_any_unknown_nodes(struct seeker *seeker)
{
	if (!seeker->unknown_nodes)
		return false;

	seeker->unknown_nodes = false;
	probe_many_random_scids(seeker);
	return true;
}

/* Periodic timer to see how our gossip is going. */
static void seeker_check(struct seeker *seeker)
{
	/* We don't do anything until we're synced. */
	if (seeker->daemon->current_blockheight == 0)
		goto out;

	switch (seeker->state) {
	case STARTING_UP:
		check_firstpeer(seeker);
		break;
	case PROBING_SCIDS:
		check_probe(seeker, peer_gossip_probe_scids);
		break;
	case ASKING_FOR_UNKNOWN_SCIDS:
		check_probe(seeker, probe_many_random_scids);
		break;
	case ASKING_FOR_STALE_SCIDS:
		check_probe(seeker, probe_some_random_scids);
		break;
	case PROBING_NANNOUNCES:
		check_probe(seeker, peer_gossip_probe_nannounces);
		break;
	case NORMAL:
		maybe_get_new_peer(seeker);
		maybe_rotate_gossipers(seeker);
		if (!seek_any_unknown_scids(seeker)
		    && !seek_any_stale_scids(seeker))
			seek_any_unknown_nodes(seeker);
		break;
	}

out:
	begin_check_timer(seeker);
}

/* Mutual recursion */
static void begin_sync_timer(struct seeker *seeker);

/* Periodically ask for a full sync from a random peer to backfill anything
 * we might have missed. */
static void full_sync_random_peer(struct seeker *seeker)
{
	/* Select random peer */
	struct peer *random_peer;
	struct peer_node_id_map_iter it;
	random_peer = first_random_peer(seeker->daemon, &it);
	if (!random_peer) {
		begin_sync_timer(seeker);
		return;
	}
	/* Don't repeatedly resync from the same node. */
	if (seeker->last_full_sync_peer && seeker->last_full_sync_peer == random_peer) {
		struct peer *new_peer;
		new_peer = next_random_peer(seeker->daemon, random_peer, &it);
		if (new_peer)
			random_peer = new_peer;
		else {
			begin_sync_timer(seeker);
			return;
		}
	}
	status_peer_debug(&random_peer->id,
			  "seeker: chosen for periodic full sync");
	normal_gossip_start(seeker,random_peer, true);
	seeker->last_full_sync_peer = random_peer;
	begin_sync_timer(seeker);
}

static void begin_sync_timer(struct seeker *seeker)
{
	if (seeker->sync_timer)
		tal_free(seeker->sync_timer);
	const u32 polltime = GOSSIP_SEEKER_RESYNC_INTERVAL(seeker);

	seeker->sync_timer = new_reltimer(&seeker->daemon->timers,
					  seeker,
					  time_from_sec(polltime),
					  full_sync_random_peer, seeker);
}

/* We get this when we have a new peer. */
void seeker_setup_peer_gossip(struct seeker *seeker, struct peer *peer)
{
	/* Can't do anything useful with these peers. */
	if (!peer->gossip_queries_feature)
		return;

	/* Don't start gossiping until we're synced. */
	if (seeker->daemon->current_blockheight == 0)
		return;

	switch (seeker->state) {
	case STARTING_UP:
		if (seeker->random_peer == NULL) {
			peer_gossip_startup(seeker, peer, true);
			/* Get another full gossip sync later. */
			begin_sync_timer(seeker);
		}
		/* Waiting for seeker_check to release us */
		return;

	/* In these states, we set up peers to stream gossip normally */
	case PROBING_SCIDS:
	case PROBING_NANNOUNCES:
	case NORMAL:
	case ASKING_FOR_UNKNOWN_SCIDS:
	case ASKING_FOR_STALE_SCIDS:
		normal_gossip_start(seeker, peer, false);
		return;
	}
	abort();
}

bool remove_unknown_scid(struct seeker *seeker,
			 const struct short_channel_id *scid,
			 bool found /*FIXME: use this info!*/)
{
	return uintmap_del(&seeker->unknown_scids, scid->u64);
}

/* This peer told us about an update to an unknown channel.  Ask it for a
 * channel_announcement. */
void query_unknown_channel(struct daemon *daemon,
			   const struct node_id *source_peer,
			   struct short_channel_id unknown_scid)
{
	add_unknown_scid(daemon->seeker,
			 unknown_scid,
			 source_peer ? find_peer(daemon, source_peer) : NULL);
}

/* This peer told us about an unknown node.  Start probing it. */
void query_unknown_node(struct daemon *daemon,
			const struct node_id *source_peer,
			const struct node_id *unknown_node)
{
	daemon->seeker->unknown_nodes = true;
	if (source_peer)
		set_preferred_peer(daemon->seeker,
				   find_peer(daemon, source_peer));
}

/* Peer has died, NULL out any pointers we have */
void seeker_peer_gone(struct seeker *seeker, const struct peer *peer)
{
	if (seeker->random_peer == peer)
		seeker->random_peer = NULL;

	for (size_t i = 0; i < tal_count(seeker->gossiper); i++) {
		if (seeker->gossiper[i] == peer)
			seeker->gossiper[i] = NULL;
	}

	if (seeker->preferred_peer == peer)
		seeker->preferred_peer = NULL;
}

void dev_seeker_memleak(struct htable *memtable, struct seeker *seeker)
{
	memleak_scan_uintmap(memtable, &seeker->stale_scids);
}
