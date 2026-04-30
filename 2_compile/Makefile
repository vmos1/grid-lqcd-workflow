# GRID_INSTALL, MPICXX, and OMPI_CXX are set by config.sh and exported by build_exec.sh
GRID_CONFIG = $(GRID_INSTALL)/bin/grid-config
CXX         = $(MPICXX)
CXXFLAGS    = $(shell $(GRID_CONFIG) --cxxflags)
LDFLAGS     = $(shell $(GRID_CONFIG) --ldflags)
LIBS        = $(shell $(GRID_CONFIG) --libs)

SRCDIR = src
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.cc)
BINS = $(patsubst $(SRCDIR)/%.cc,$(BINDIR)/%,$(SRCS))

all: $(BINDIR) $(BINS)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/%: $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f $(BINDIR)/*

.PHONY: all clean
