# Root Makefile — every target forwards to bre/Makefile (the Behavioral Risk Engine).
# The PHY234 rocket-simulation material at this level is unrelated and untouched.
#
# From a clean clone:
#     make setup && make sim && make recover && make fit && make eval && make dashboard

TARGETS := help setup test data sim recover fit eval api dashboard retrain instrument report clean

.PHONY: $(TARGETS)
.DEFAULT_GOAL := help

$(TARGETS):
	$(MAKE) -C bre $@

# Any target not listed above is forwarded as well, so new bre/ targets need no edit here.
.DEFAULT:
	$(MAKE) -C bre $@
