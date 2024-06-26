#ifndef LIGHTNING_COMMON_SUBDAEMON_H
#define LIGHTNING_COMMON_SUBDAEMON_H
#include "config.h"
#include <common/daemon.h>

struct htable;

/* daemon_setup, but for subdaemons: returns true if --developer */
bool subdaemon_setup(int argc, char *argv[]);

#endif /* LIGHTNING_COMMON_SUBDAEMON_H */
