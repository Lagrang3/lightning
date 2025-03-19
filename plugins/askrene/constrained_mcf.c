#include "config.h"
#include <assert.h>
#include <ccan/asort/asort.h>
#include <ccan/bitmap/bitmap.h>
#include <ccan/list/list.h>
#include <ccan/tal/str/str.h>
#include <ccan/tal/tal.h>
#include <common/utils.h>
#include <float.h>
#include <math.h>
#include <plugins/askrene/algorithm.h>
#include <plugins/askrene/askrene.h>
#include <plugins/askrene/constrained_mcf.h>
#include <plugins/askrene/dijkstra.h>
#include <plugins/askrene/explain_failure.h>
#include <plugins/askrene/flow.h>
#include <plugins/askrene/graph.h>
#include <plugins/askrene/refine.h>
#include <plugins/libplugin.h>
#include <stdint.h>

/* OPTIMAL PAYMENTS
 *
 * The routing of a payment is modeled as a constrained optimization problem.
 * We wish to maximize the probability of success of the routing while keeping
 * the total fees and *total delay under certain constraint values. To do this
 * we construct three cost functions: probability, fee and delay, with a
 * proportional and constant components and then solve a constrained minimum
 * cost flow problem using an **approximate algorithm.
 *
 * *as a matter of fact is the maximum delay over all routes that one wishes to
 * constrain but we are not there yet and we instead constrain the delay by
 * putting a bound on the sum of all delays deltas for each hop.
 *
 * **an exact solution would require an exponential runtime.
 *
 *
 * FEE COST
 *
 * Routing fees is non-linear function of the payment flow x, that's true even
 * without the base fee:
 *
 * 	fee_msat = base_msat + floor(millionths*x_msat / 10^6)
 *
 * We approximate this fee into a linear function plus a constant term:
 *
 * 	cost(x) = k*base_msat + k * A_msat * millionths/10^6 * x
 *
 * where A_msat is the accuracy of the flows, the smallest unit of flow, so that
 * x_mast = x * A_msat and k is an arbitrary constant.
 * A channel with a cost PPM such that the entire payment amount T would lead to
 * a 0msat fee can be discarded as 0 in the linear approximation.
 * That is, choosing `k = 2 T_msat/A_msat` the slope of the linear approximation
 * becomes
 *
 *      M = 2 * T_msat * millionths / 10^6
 *
 * when costs per unit of flow are integers this term is neglected for those
 * channels that would have paid 0 fees for the entire payment amount. The
 * constant factor then becomes
 *
 *      h = k*base_msat = 2 * T_msat / A_msat * base_msat
 *
 * To compute the fee_msat from the linearized cost we would compute
 *
 *      fee_msat = x*M/k = x * A_msat * millionths / 10^6
 *
 * and the fee_msat from the constant cost
 *
 *      fee_msat = h / k = base_msat
 *
 *
 * PROBABILITY COST
 *
 * The probability of success P of the payment is the product of the prob. of
 * success of forwarding parts of the payment over all routing channels. This
 * problem is separable if we log it, and since we would like to increase P,
 * then we can seek to minimize -log(P), and that's our prob. cost function [1].
 *
 * 	- log P = sum_{i} - log P_i
 *
 * The probability of success `P_i` of sending some flow `x` on a channel with
 * liquidity l in the range a<=l<b is
 *
 * 	P_{a,b}(x) = (b-x)/(b-a); for x > a
 * 		   = 1.         ; for x <= a
 *
 * Notice that unlike the similar formula in [1], the one we propose does not
 * contain the quantization shot noise for counting states.
 *
 * The cost associated to probability P is then -k log P, where k is some
 * constant. For k=1 we get the following table:
 *
 * 	prob | cost
 * 	-----------
 * 	0.01 | 4.6
 * 	0.02 | 3.9
 * 	0.05 | 3.0
 * 	0.10 | 2.3
 * 	0.20 | 1.6
 * 	0.50 | 0.69
 * 	0.80 | 0.22
 * 	0.90 | 0.10
 * 	0.95 | 0.05
 * 	0.98 | 0.02
 * 	0.99 | 0.01
 *
 * Clearly -log P(x) is non-linear; we try to linearize it piecewise:
 * split the channel into 5 arcs representing 5 liquidity regions:
 *
 * 	arc_0 -> [0, a)
 * 	arc_1 -> [a, a+(b-a)*f1)
 * 	arc_2 -> [a+(b-a)*f1, a+(b-a)*f2)
 * 	arc_3 -> [a+(b-a)*f2, a+(b-a)*f3)
 * 	arc_4 -> [a+(b-a)*f3, a+(b-a)*f4)
 *
 * where f1 = 0.5, f2 = 0.8, f3 = 0.95, f4 = 0.99;
 * We fill arc_0's capacity with complete certainty P=1, then if more flow is
 * needed we start filling the capacity in arc_1 until the total probability
 * of success reaches P=0.5, then arc_2 until P=1-0.8=0.2, then arc_3
 * P=1-0.95=0.05 and finally arc_4 with P=1-0.99=0.01.
 * We don't go further than 1% prob. of success per channel.
 * With this choice, the slope of the linear cost function becomes:
 *
 * 	m_0 = 0
 * 	m_1 = 1.38 k/(b-a)
 * 	m_2 = 3.05 k/(b-a)
 * 	m_3 = 9.24 k/(b-a)
 * 	m_4 = 40.2 k/(b-a)
 *
 * Notice that one of the assumptions in [2] for the MCF problem is that flows
 * and the slope of the costs functions are integer numbers. The only way we
 * have at hand to make it so, is to choose a universal value of `k` that scales
 * up the slopes so that floor(m_i) is not zero for every arc.
 * We will use k = 1000T where T is the payment amount, that is to say we
 * neglect (make it zero) for channel parts whose probability of failing on the
 * entire payment amount is less than 0.1%.
 *
 *
 * CONSTANT PROBABILITY COST
 *
 * We assume that for any channel chosen at random will have some probability of
 * failure for other causes other than liquidity then our probability of success
 * will have two components: a constant factor and function of the channel
 * liquidity.
 *
 *      P_{a,b}(x) = P_const * P_convex(x;a,b)
 *
 * P_convex(x;a,b) is the prability function that we linearized in the previous
 * section. Now we have to deal with a cost function that has a constant term:
 *
 *     cost(x) = h + cost_convex(x;a,b)
 *
 * where h = - k log P_const, and cost_convex(x;a,b,) = - k log P_convex(x;a,b).
 * To take the constant term in the linearization of the cost function we split
 * every channel into a 6 parts, 5 of which follow the convex cost linearization
 * explained in the previous section and a final part that carries only the
 * constant term. A channel that connects nodes A and B:
 *
 *      (A) ---> (B)
 *
 * becomes after linearization:
 *
 *                   (multiple parallel channels)
 *           h          m_0,m_1,...
 *      (A) ---> (C) ------------> (B)
 *
 * where C is an auxiliary node. The arc A->C has zero proportional cost and a
 * fixed cost h for any amount of flow, for a constant probability of success
 * of:
 *      Prob = exp(-h/k)
 *
 * The i-th arc between C->B has zero fixed cost but proportional cost of
 * M_i = m_i*k/(b-a), for a probability of success of forwarding x units of
 * flow:
 *
 *      Prob = exp(-M_i*x/k) = exp( -m_i*x/(b-a) )
 *
 *
 * REFERENCES
 *
 * [1] Pickhardt and Richter, https://arxiv.org/abs/2107.05322
 * [2] R.K. Ahuja, T.L. Magnanti, and J.B. Orlin. Network Flows:
 * Theory, Algorithms, and Applications. Prentice Hall, 1993.
 *
 * */

/* Up to 8 parts per channel, in fact we use 6 */
#define PARTS_BITS 3
/* How many parts correspond to the convex cost linearization.
 * The part indexes run from 0 to N_LINEAR_PARTS-1 */
#define N_LINEAR_PARTS 5
/* Value of the part that takes care to the base costs. */
#define CHANNEL_BASE_PART 7

/* These are the probability intervals we use to decompose a channel into linear
 * cost function arcs. */
static const double CHANNEL_PIVOTS[] = {0, 0.5, 0.8, 0.95, 0.99};

static const s64 INFINITE = INT64_MAX;

/* Let's try this encoding of arcs:
 * FIXME: update explanation here
 * Each channel `c` has two possible directions identified by a bit
 * `half` or `!half`, and each one of them has to be
 * decomposed into 6 liquidity parts in order to
 * linearize the cost function, but also to solve MCF
 * problem we need to keep track of flows in the
 * residual network hence we need for each directed arc
 * in the network there must be another arc in the
 * opposite direction refered to as it's dual. In total
 * 1+3+1 additional bits of information:
 *
 * 	(chan_idx)(half)(part)(dual)
 *
 * That means, for each channel we need to store the
 * information of 32 arcs. If we implement a convex-cost
 * solver then we can reduce that number to size(half)size(dual)=4.
 *
 * In the adjacency of a `node` we are going to store
 * the outgoing arcs. If we ever need to loop over the
 * incoming arcs then we will define a reverse adjacency
 * API.
 * Then for each outgoing channel `(c,half)` there will
 * be 6 (3 bits) parts for the actual residual capacity, hence
 * with the dual bit set to 0:
 *
 * 	(c,half,0,0)
 * 	(c,half,1,0)
 * 	(c,half,2,0)
 * 	(c,half,3,0)
 *
 * and also we need to consider the dual arcs
 * corresponding to the channel direction `(c,!half)`
 * (the dual has reverse direction):
 *
 * 	(c,!half,0,1)
 * 	(c,!half,1,1)
 * 	(c,!half,2,1)
 * 	(c,!half,3,1)
 *
 * These are the 8 outgoing arcs relative to `node` and
 * associated with channel `c`. The incoming arcs will
 * be:
 *
 * 	(c,!half,0,0)
 * 	(c,!half,1,0)
 * 	(c,!half,2,0)
 * 	(c,!half,3,0)
 *
 * 	(c,half,0,1)
 * 	(c,half,1,1)
 * 	(c,half,2,1)
 * 	(c,half,3,1)
 *
 * but they will be stored as outgoing arcs on the peer
 * node `next`.
 * */

/*
 * We want to use the whole number here for convenience, but
 * we can't us a union, since bit order is implementation-defined and
 * we want chanidx on the highest bits:
 *
 * [ 0       1 2 3   4            5 6 ... 31 ]
 *   dual    part    chandir      chanidx
 */
#define ARC_DUAL_BITOFF (0)
#define ARC_PART_BITOFF (1)
#define ARC_CHANDIR_BITOFF (1 + PARTS_BITS)
#define ARC_CHANIDX_BITOFF (1 + PARTS_BITS + 1)
#define ARC_CHANIDX_BITS  (32 - ARC_CHANIDX_BITOFF)

/* How many arcs can we have for a single channel?
 * linearization parts, both directions, and dual */
#define ARCS_PER_CHANNEL ((size_t)1 << (PARTS_BITS + 1 + 1))

static inline void arc_to_parts(struct arc arc,
				u32 *chanidx,
				int *chandir,
				u32 *part,
				bool *dual)
{
	if (chanidx)
		*chanidx = (arc.idx >> ARC_CHANIDX_BITOFF);
	if (chandir)
		*chandir = (arc.idx >> ARC_CHANDIR_BITOFF) & 1;
	if (part)
		*part = (arc.idx >> ARC_PART_BITOFF) & ((1 << PARTS_BITS)-1);
	if (dual)
		*dual = (arc.idx >> ARC_DUAL_BITOFF) & 1;
}

static inline struct arc arc_from_parts(u32 chanidx, int chandir, u32 part, bool dual)
{
	struct arc arc;
	assert(part < (1 << PARTS_BITS));
	assert(chandir == 0 || chandir == 1);
	assert(chanidx < (1U << ARC_CHANIDX_BITS));
	arc.idx = ((u32)dual << ARC_DUAL_BITOFF)
		| (part << ARC_PART_BITOFF)
		| ((u32)chandir << ARC_CHANDIR_BITOFF)
		| (chanidx << ARC_CHANIDX_BITOFF);
	return arc;
}

/* ID of the auxiliary node use to separate channel base costs from proportional
 * cost. */
static u32 auxiliary_node(u32 chanidx, int chandir, u32 auxiliary_node_offset)
{
	return ((chanidx << 1) | chandir) + auxiliary_node_offset;
}

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static void write_pickhardt_richter_templates(double *cap_fraction,
					      double *cost_fraction)
{
	/* This is Pickhardt-Richter's probability cost function. */
	cap_fraction[0] = 0;
	cost_fraction[0] = 0;
	for (size_t i = 1; i < N_LINEAR_PARTS; i++) {
		cap_fraction[i] = CHANNEL_PIVOTS[i] - CHANNEL_PIVOTS[i - 1];
		cost_fraction[i] =
		    log((1 - CHANNEL_PIVOTS[i - 1]) / (1 - CHANNEL_PIVOTS[i])) /
		    cap_fraction[i];
	}
}

struct pay_parameters {
	const struct route_query *rq;
        
	/* full payment amount */
	struct amount_msat amount;
	
        /* base unit for computation, ie. accuracy */
	struct amount_msat accuracy;
	
        /* channel linearization template */
	double cap_fraction[N_LINEAR_PARTS],
	       cost_fraction[N_LINEAR_PARTS];
        
        u32 auxiliary_node_offset;
        double prob_scale_factor, fee_scale_factor;
        double constant_probability_fail;
        
        const struct gossmap_node *source;
	const struct gossmap_node *target;
};

static struct graph *build_graph(const tal_t *ctx, struct route_query *rq)
{
	const size_t max_num_chans = gossmap_max_chan_idx(rq->gossmap);
	const size_t max_num_nodes = gossmap_max_node_idx(rq->gossmap);

	const size_t max_num_arcs = max_num_chans * ARCS_PER_CHANNEL;
	const size_t max_num_edges = max_num_nodes + max_num_chans * 2;

	struct graph *graph =
	    graph_new(ctx, max_num_edges, max_num_arcs, ARC_DUAL_BITOFF);
	return graph;
}

/* Set *capacity to value, up to *cap_on_capacity.  Reduce cap_on_capacity */
static void set_capacity(s64 *capacity, u64 value, u64 *cap_on_capacity)
{
	*capacity = MIN(value, *cap_on_capacity);
	*cap_on_capacity -= *capacity;
}

/* Split a directed channel into parts with linear cost function. */
static void linearize_channel_prob_cost(const struct pay_parameters *params,
					const struct gossmap_chan *c,
					const int dir, s64 *capacity, s64 *cost)
{
	struct amount_msat mincap, maxcap;

	/* This takes into account any payments in progress. */
	get_constraints(params->rq, c, dir, &mincap, &maxcap);

	/* Assume if min > max, min is wrong */
	if (amount_msat_greater(mincap, maxcap))
		mincap = maxcap;

	u64 a = amount_msat_ratio_floor(mincap, params->accuracy),
	    b = 1 + amount_msat_ratio_floor(maxcap, params->accuracy);

	/* An extra bound on capacity, here we use it to reduce the flow such
	 * that it does not exceed htlcmax. */
	u64 cap_on_capacity = amount_msat_ratio_floor(
	    gossmap_chan_htlc_max(c, dir), params->accuracy);

	set_capacity(&capacity[0], a, &cap_on_capacity);
	cost[0] = 0;
	for (size_t i = 1; i < N_LINEAR_PARTS; i++) {
		set_capacity(&capacity[i], params->cap_fraction[i] * (b - a),
			     &cap_on_capacity);
		/* We set
		 *      cost_per_unit_flow = cost_fraction * prob_scale_factor/(b-a),
		 *
		 * cost_fraction: is the template slope (cost per unit of
		 * probability) prob_scale_factor: 1000 * total_payment_amount
		 *
		 * That means a linear part that would give a fail probability
		 * of less than 0.1% for the entire payment amount has a
		 * negligible cost. */
		cost[i] = params->cost_fraction[i] * params->prob_scale_factor /
			  (b - a);
	}
}

static u64 sum_capacity(const s64 *capacity)
{
	u64 r = 0;
	for (size_t i = 0; i < N_LINEAR_PARTS; i++)
		r += capacity[i];
	return r;
}

/* Extract the capacity, probability and fee costs for channels in the gossmap.
 */
static void linearize_network(struct pay_parameters *params,
			      struct graph *graph, s64 *capacity,
			      s64 *prob_cost, s64 *base_prob_cost,
			      s64 *fee_cost, s64 *base_fee_cost,
			      s64 *delay_cost, s64 *base_delay_cost)
{
	s64 channel_capacity[N_LINEAR_PARTS], channel_prob_cost[N_LINEAR_PARTS];
	const double ln_30 = log(30);

	const struct gossmap *gossmap = params->rq->gossmap;
	for (struct gossmap_node *node = gossmap_first_node(gossmap); node;
	     node = gossmap_next_node(gossmap, node)) {
		const u32 node_id = gossmap_node_idx(gossmap, node);
		for (size_t j = 0; j < node->num_chans; j++) {
			int half;
			const struct gossmap_chan *c =
			    gossmap_nth_chan(gossmap, node, j, &half);
			/* a channel disabled from above */
			if (!gossmap_chan_set(c, half) ||
			    !c->half[half].enabled)
				continue;

			const u32 chan_id = gossmap_chan_idx(gossmap, c);
			const struct gossmap_node *next =
			    gossmap_nth_node(gossmap, c, !half);
			const u32 next_id = gossmap_node_idx(gossmap, next);

			/* a self-channel? */
			if (node_id == next_id)
				continue;

			/* Construct the linear arcs middle_node->next_node */
			struct arc arc, dual;
			const u32 middle_node = auxiliary_node(
			    chan_id, half, params->auxiliary_node_offset);

			linearize_channel_prob_cost(params, c, half,
						    channel_capacity,
						    channel_prob_cost);

			/* linear fee_cost per unit of flow */
			const s64 channel_fee =
			    2 * c->half[half].proportional_fee * 1e-6 *
			    params->amount
				.millisatoshis; /* raw: bump up the slope */

			/* Same bias factor defined by Rusty.
			 * FIXME: This choice is a bit arbitrary though it has a
			 * nice stacking property.
			 *
			 *      bias_factor(b1+b2) =
			 *              bias_factor(b1) * bias_factor(b2)
			 *
			 * It would be interesting to define a different bias
			 * function that has a clear probabilistic
			 * interpretation. */
			u8 bias = params->rq->biases[(chan_id << 1) | half];
			double bias_factor = 1;
			if (bias)
				bias_factor = exp(-bias / (100 / ln_30));

			for (size_t k = 0; k < N_LINEAR_PARTS; k++) {
				arc = arc_from_parts(chan_id, half, k, false);
				graph_add_arc(graph, arc, node_obj(middle_node),
					      node_obj(next_id));

				capacity[arc.idx] = channel_capacity[k];

				prob_cost[arc.idx] =
				    channel_prob_cost[k] * bias_factor;
				base_prob_cost[arc.idx] = 0;

				fee_cost[arc.idx] = channel_fee;
				base_fee_cost[arc.idx] = 0;

				delay_cost[arc.idx] = 0;
				base_delay_cost[arc.idx] = 0;

				// + the respective dual
				dual = arc_dual(graph, arc);

				capacity[dual.idx] = 0;

				prob_cost[dual.idx] = -prob_cost[arc.idx];
				base_prob_cost[dual.idx] = 0;

				fee_cost[dual.idx] = -fee_cost[arc.idx];
				base_fee_cost[dual.idx] = 0;

				delay_cost[arc.idx] = 0;
				base_delay_cost[arc.idx] = 0;
			}

			/* Construct the base cost arc: start_node->middle_node
			 */
			arc = arc_from_parts(chan_id, half, CHANNEL_BASE_PART,
					     false);
			graph_add_arc(graph, arc, node_obj(node_id),
				      node_obj(middle_node));

			capacity[arc.idx] = sum_capacity(channel_capacity);

			prob_cost[arc.idx] = 0;
			base_prob_cost[arc.idx] =
			    params->prob_scale_factor *
			    (-log(1 - params->constant_probability_fail)) *
			    bias_factor;

			fee_cost[arc.idx] = 0;
			base_fee_cost[arc.idx] =
			    c->half[half].base_fee * params->fee_scale_factor;

			delay_cost[arc.idx] = 0;
			base_delay_cost[arc.idx] = c->half[half].delay;

			// + the respective dual
			dual = arc_dual(graph, arc);
			capacity[dual.idx] = 0;

			prob_cost[dual.idx] = 0;
			base_prob_cost[dual.idx] = -base_prob_cost[arc.idx];

			fee_cost[dual.idx] = 0;
			base_fee_cost[dual.idx] = -base_fee_cost[arc.idx];

			delay_cost[dual.idx] = 0;
			base_delay_cost[dual.idx] = -base_delay_cost[arc.idx];
		}
	}
}

/* Helper function.
 * Given an arc give me the flow. */
static s64 get_arc_flow(const struct graph *graph, const s64 *capacity,
			const struct arc arc)
{
	assert(!arc_is_dual(graph, arc));
	struct arc dual = arc_dual(graph, arc);
	assert(dual.idx < tal_count(capacity));
	return capacity[dual.idx];
}

/* flow on directed channels */
struct chan_flow {
	s64 half[2];
};

/* Search in the network a path of positive flow until we reach a node with
 * positive balance (returns a node idx with positive balance)
 * or we discover a cycle (returns a node idx with 0 balance).
 * */
static u32 find_path_or_cycle(const tal_t *ctx, const struct gossmap *gossmap,
			      const struct chan_flow *chan_flow,
			      const u32 source, const s64 *balance,
			      const struct gossmap_chan **prev_chan,
			      int *prev_dir, u32 *prev_idx)
{
	const tal_t *working_ctx = tal(ctx, tal_t);
	const size_t max_num_nodes = gossmap_max_node_idx(gossmap);
	bitmap *visited =
	    tal_arrz(working_ctx, bitmap, BITMAP_NWORDS(max_num_nodes));
	u32 final_idx = source;
	bitmap_set_bit(visited, final_idx);

	/* It is guaranteed to halt, because we either find a node with
	 * balance[]>0 or we hit a node twice and we stop. */
	while (balance[final_idx] <= 0) {
		u32 updated_idx = INVALID_INDEX;
		struct gossmap_node *cur =
		    gossmap_node_byidx(gossmap, final_idx);

		for (size_t i = 0; i < cur->num_chans; i++) {
			int dir;
			const struct gossmap_chan *c =
			    gossmap_nth_chan(gossmap, cur, i, &dir);

			if (!gossmap_chan_set(c, dir) || !c->half[dir].enabled)
				continue;

			const u32 c_idx = gossmap_chan_idx(gossmap, c);

			/* follow the flow */
			if (chan_flow[c_idx].half[dir] > 0) {
				const struct gossmap_node *n =
				    gossmap_nth_node(gossmap, c, !dir);
				u32 next_idx = gossmap_node_idx(gossmap, n);

				prev_dir[next_idx] = dir;
				prev_chan[next_idx] = c;
				prev_idx[next_idx] = final_idx;

				updated_idx = next_idx;
				break;
			}
		}

		assert(updated_idx != INVALID_INDEX);
		assert(updated_idx != final_idx);
		final_idx = updated_idx;

		if (bitmap_test_bit(visited, updated_idx)) {
			/* We have seen this node before, we've found a cycle.
			 */
			assert(balance[updated_idx] <= 0);
			break;
		}
		bitmap_set_bit(visited, updated_idx);
	}
	tal_free(working_ctx);
	return final_idx;
}

/* Given a path from a node with negative balance to a node with positive
 * balance, compute the bigest flow and substract it from the nodes balance and
 * the channels allocation. */
static struct flow *substract_flow(const tal_t *ctx,
				   const struct pay_parameters *params,
				   const u32 source, const u32 sink,
				   s64 *balance, struct chan_flow *chan_flow,
				   const u32 *prev_idx, const int *prev_dir,
				   const struct gossmap_chan *const *prev_chan)
{
	const struct gossmap *gossmap = params->rq->gossmap;
	assert(balance[source] < 0);
	assert(balance[sink] > 0);
	s64 delta = MIN(-balance[source], balance[sink]);
	size_t length = 0;

	/* We can only walk backwards, now get me the legth of the path and the
	 * max flow we can send through this route. */
	for (u32 cur_idx = sink; cur_idx != source;
	     cur_idx = prev_idx[cur_idx]) {
		assert(cur_idx != INVALID_INDEX);
		const int dir = prev_dir[cur_idx];
		const struct gossmap_chan *const chan = prev_chan[cur_idx];

		/* we could optimize here by caching the idx of the channels in
		 * the path, but the bottleneck of the algorithm is the MCF
		 * computation not here. */
		const u32 chan_idx = gossmap_chan_idx(gossmap, chan);

		delta = MIN(delta, chan_flow[chan_idx].half[dir]);
		length++;
	}

	struct flow *f = tal(ctx, struct flow);
	f->path = tal_arr(f, const struct gossmap_chan *, length);
	f->dirs = tal_arr(f, int, length);

	/* Walk again and substract the flow value (delta). */
	assert(delta > 0);
	balance[source] += delta;
	balance[sink] -= delta;
	for (u32 cur_idx = sink; cur_idx != source;
	     cur_idx = prev_idx[cur_idx]) {
		const int dir = prev_dir[cur_idx];
		const struct gossmap_chan *const chan = prev_chan[cur_idx];
		const u32 chan_idx = gossmap_chan_idx(gossmap, chan);

		length--;
		/* f->path and f->dirs contain the channels in the path in the
		 * correct order. */
		f->path[length] = chan;
		f->dirs[length] = dir;

		chan_flow[chan_idx].half[dir] -= delta;
	}
	if (!amount_msat_mul(&f->delivers, params->accuracy, delta))
		abort();
	return f;
}

/* Substract a flow cycle from the channel allocation. */
static void substract_cycle(const struct gossmap *gossmap, const u32 sink,
			    struct chan_flow *chan_flow, const u32 *prev_idx,
			    const int *prev_dir,
			    const struct gossmap_chan *const *prev_chan)
{
	s64 delta = INFINITE;
	u32 cur_idx;

	/* Compute greatest flow in this cycle. */
	for (cur_idx = sink;; cur_idx = prev_idx[cur_idx]) {
		assert(cur_idx != INVALID_INDEX);
		const int dir = prev_dir[cur_idx];
		const struct gossmap_chan *const chan = prev_chan[cur_idx];
		const u32 chan_idx = gossmap_chan_idx(gossmap, chan);

		delta = MIN(delta, chan_flow[chan_idx].half[dir]);

		if (cur_idx == sink)
			/* we have come back full circle */
			break;
	}
	assert(cur_idx == sink);

	/* Walk again and substract the flow value (delta). */
	assert(delta < INFINITE);
	assert(delta > 0);

	for (cur_idx = sink;; cur_idx = prev_idx[cur_idx]) {
		assert(cur_idx != INVALID_INDEX);
		const int dir = prev_dir[cur_idx];
		const struct gossmap_chan *const chan = prev_chan[cur_idx];
		const u32 chan_idx = gossmap_chan_idx(gossmap, chan);

		chan_flow[chan_idx].half[dir] -= delta;

		if (cur_idx == sink)
			/* we have come back full circle */
			break;
	}
	assert(cur_idx == sink);
}

/* Given a flow in the residual network, build a set of payment flows in the
 * gossmap that corresponds to this flow. */
static struct flow **get_flow_paths(const tal_t *ctx,
				    const struct pay_parameters *params,
				    struct graph *graph, s64 *capacity)
{
	struct flow **flows = tal_arr(ctx, struct flow *, 0);
	const tal_t *working_ctx = tal(ctx, tal_t);

	const struct gossmap *gossmap = params->rq->gossmap;

	/* We use the gossmap nodes instead of graph nodes! */
	const size_t max_num_nodes = gossmap_max_node_idx(gossmap);
	const size_t max_num_chans = gossmap_max_chan_idx(gossmap);

	/* balance of flow in and out of nodes. We build routes by finding
	 * directed paths from nodes with negative balance (net sources) to
	 * nodes with positive balance (net sinks). */
	s64 *balance = tal_arrz(working_ctx, s64, max_num_nodes);
	struct chan_flow *chan_flow =
	    tal_arrz(working_ctx, struct chan_flow, max_num_chans);

	/* Auxiliary data to reconstruct a path source->sink once found. */
	const struct gossmap_chan **prev_chan =
	    tal_arr(working_ctx, const struct gossmap_chan *, max_num_nodes);
	int *prev_dir = tal_arr(working_ctx, int, max_num_nodes);
	u32 *prev_idx = tal_arr(working_ctx, u32, max_num_nodes);

	for (u32 node_idx = 0; node_idx < max_num_nodes; node_idx++)
		prev_idx[node_idx] = INVALID_INDEX;

	/* gossmap node i maps to graph node i, the reverse is not true */
	for (struct node n = {.idx = 0}; n.idx < max_num_nodes; n.idx++) {
		for (struct arc arc = node_adjacency_begin(graph, n);
		     !node_adjacency_end(arc);
		     arc = node_adjacency_next(graph, arc)) {
			if (arc_is_dual(graph, arc))
				continue;

			int chandir;
			u32 chanidx, chanpart;
			arc_to_parts(arc, &chanidx, &chandir, &chanpart, NULL);

			/* The flow on the base part equals the sum of the flows
			 * on the proportional parts. Also arcs exiting gossmap
			 * nodes are always base arcs. */
			assert(chanpart == CHANNEL_BASE_PART);

			struct gossmap_chan *chan =
			    gossmap_chan_byidx(gossmap, chanidx);
			const struct gossmap_node *next =
			    gossmap_nth_node(gossmap, chan, !chandir);
			const u32 next_id = gossmap_node_idx(gossmap, next);

			/* check we did not confuse the direction */
			assert(n.idx != next_id);

			s64 flow_value = get_arc_flow(graph, capacity, arc);
			balance[n.idx] -= flow_value;
			balance[next_id] += flow_value;
			chan_flow[chanidx].half[chandir] += flow_value;
		}
	}

	/* Select all nodes with negative balance and find a flow that reaches a
	 * positive balance node. */
	for (u32 source = 0; source < max_num_nodes; source++) {
		// this node has negative balance, flows leaves from here
		while (balance[source] < 0) {
			prev_chan[source] = NULL;
			u32 sink = find_path_or_cycle(
			    working_ctx, gossmap, chan_flow, source, balance,
			    prev_chan, prev_dir, prev_idx);

			if (balance[sink] > 0)
			/* case 1. found a path */
			{
				struct flow *fp = substract_flow(
				    flows, params, source, sink, balance,
				    chan_flow, prev_idx, prev_dir, prev_chan);

				tal_arr_expand(&flows, fp);
			} else
			/* case 2. found a cycle */
			{
                                /* I don't think the current MCF algorithms I
                                 * have implemented can produce cycles. If one
                                 * is found we log it for further inspection. */
				rq_log(tmpctx, params->rq, LOG_UNUSUAL,
				       "%s: found a cycle", __func__);
				substract_cycle(gossmap, sink, chan_flow,
						prev_idx, prev_dir, prev_chan);
			}
		}
	}

	tal_free(working_ctx);
	return flows;
}

/* An experimental MCF solver with extra constraints. */
const char *constrained_mcf_routes(const tal_t *ctx, struct route_query *rq,
				   const struct gossmap_node *source,
				   const struct gossmap_node *target,
				   struct amount_msat amount,
				   struct amount_msat maxfee, u32 maxdelay,
				   struct flow ***flows, double *probability)
{
	const char *ret = NULL;
	const tal_t *working_ctx = tal(ctx, tal_t);
	
        struct pay_parameters *params
                = tal(working_ctx, struct pay_parameters);
        params->rq = rq;
        params->amount = amount;
        params->accuracy = 
	    amount_msat_max(amount_msat_div(amount, 1000), AMOUNT_MSAT(1));
	params->source = source;
	params->target = target;
	params->auxiliary_node_offset = gossmap_max_node_idx(rq->gossmap);
	params->fee_scale_factor =
	    2 * amount_msat_ratio(params->amount, params->accuracy);
	params->prob_scale_factor =
	    1000 * amount_msat_ratio(params->amount, params->accuracy);
        /* Assume that 1% of the channels at any given moment may fail the
         * payment regardless of the amount. */
        params->constant_probability_fail = 0.01;
        /* Only solution with more than 0.1% probability of success will be
         * accepted. This bound is actually not important to us, so any very
         * small number here should be sufficient. */
        const double minimum_probability = 0.001;
        
        /* Parameters we can tune in the approximate solver to trade efficiency
         * vs accuracy. */
        const double tolerance = 0.2;
        const size_t max_num_iteratons = 100;

	write_pickhardt_richter_templates(params->cap_fraction,
					  params->cost_fraction);

	/* build the problem's graph */
        struct graph *graph = build_graph(working_ctx, rq);
	
        const size_t num_constraints = 3 /* probability, fee, delay */;
	s64 *capacity, **cost, **charge, *bounds, *excess;
	
        const size_t max_num_arcs = graph_max_num_arcs(graph);
        const size_t max_num_nodes = graph_max_num_nodes(graph);
        
	capacity = tal_arrz(graph, s64, max_num_arcs);
	cost = tal_arrz(graph, s64*, num_constraints);
	charge = tal_arrz(graph, s64*, num_constraints);
	bounds = tal_arrz(graph, s64, num_constraints);
	excess = tal_arrz(graph, s64, max_num_nodes);
	for (size_t i = 0; i < num_constraints; i++) {
		cost[i] = tal_arrz(cost, s64, max_num_arcs);
		charge[i] = tal_arrz(charge, s64, max_num_arcs);
	}

	linearize_network(
	    params, graph, capacity,
	    /* probability -> */ cost[0], charge[0],
	    /* fee -> */ cost[1], charge[1],
	    /* delay -> */ cost[2], charge[2]);
        
        // FIXME: review these bounds, why do we not use accuraccy here?
        /* Maximum probability cost admissible. */
	bounds[0] =
	    params->prob_scale_factor * (-log(minimum_probability));
	/* Maximum fee cost admissible */
	bounds[1] = params->fee_scale_factor * maxfee.millisatoshis; /* raw: */
        /* Maximum delay cost admissible */
	bounds[2] = maxdelay;
        
	const u32 src = gossmap_node_idx(rq->gossmap, source);
	const u32 dst = gossmap_node_idx(rq->gossmap, target);
	
	/* Since we have constraint accuracy, ask to find a payment solution
	 * that can pay a bit more than the actual value rather than undershoot it.
	 * That's why we use the ceil function here. */
	const u64 pay_amount =
	    amount_msat_ratio_ceil(params->amount, params->accuracy);
        
        excess[src] = pay_amount;
	excess[dst] = -pay_amount;

	/* Notice this function is agnostic about the source and
	 * destination. Flow balance constraints are encoded into 'excess',
	 * therefore we are able to represent a problem with multiple sources
	 * and destinations. */
	if (!solve_constrained_fcnfp(
		working_ctx,
                graph,
                excess,
                capacity,
                num_constraints,
                cost,
                charge,
                bounds,
                tolerance,
                max_num_iteratons))
                {
		// FIXME: what should we log and what should we return?
		ret = rq_log(tmpctx, rq, LOG_BROKEN,
			     "%s: failed to find a constrained MCF solution",
			     __func__);
		goto fail;
	}

	/* We dissect the solution of the MCF into payment routes.
	 * Actual amounts considering fees are computed for every
	 * channel in the routes. */
	*flows = get_flow_paths(ctx, params, graph, capacity);
	if(!*flows){
		ret = rq_log(tmpctx, rq, LOG_BROKEN,
		       "%s: failed to extract flow paths from the MCF solution",
		       __func__);
		goto fail;
	}
	
	// FIXME: do we assume all constraints are valid?
	
	ret = refine_with_fees_and_limits(ctx, rq, amount, flows, probability);
	if (ret)
		goto fail;

	// FIXME: tal steal flows to ctx
	tal_free(working_ctx);
	return NULL;
fail:
	// all flows are freed
	tal_free(working_ctx);
	return ret;
}
