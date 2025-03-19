#ifndef LIGHTNING_PLUGINS_ASKRENE_CONSTRAINED_MCF_H
#define LIGHTNING_PLUGINS_ASKRENE_CONSTRAINED_MCF_H

#include "config.h"
#include <plugins/askrene/flow.h>

/* An experimental MCF solver with extra constraints. */
const char *constrained_mcf_routes(const tal_t *ctx, struct route_query *rq,
				   const struct gossmap_node *source,
				   const struct gossmap_node *target,
				   struct amount_msat amount,
				   struct amount_msat maxfee, u32 maxdelay,
				   struct flow ***flows, double *probability);

#endif /* LIGHTNING_PLUGINS_ASKRENE_CONSTRAINED_MCF_H */
