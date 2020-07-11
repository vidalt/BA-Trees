all : bornAgain

CCC = g++
CCOPT = -O3
CCFLAGS = $(CCOPT)
TARGETDIR=.

OBJS2 = \
        $(TARGETDIR)/main.o \
        $(TARGETDIR)/BornAgainDecisionTree.o \
        $(TARGETDIR)/FSpace.o \
        $(TARGETDIR)/RandomForest.o

ifdef withCPLEX
    SYSTEM        = x86-64_linux
    LIBFORMAT     = static_pic
    CPLEXVERSION  = CPLEX_Studio1210
    CPLEX_HOME    = /opt/ibm/ILOG/$(CPLEXVERSION)

    #configuring CPLEX
    CPLEXDIR      = $(CPLEX_HOME)/cplex
    CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
    CPLEXINCDIR   = $(CPLEXDIR)/include

    #configuring CONCERT
    CONCERTDIR    = $(CPLEX_HOME)/concert
    CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
    CONCERTINCDIR = $(CONCERTDIR)/include

    #updating flags
    CCOPT += -w -DIL_STD -DUSING_CPLEX
    CCFLAGS = $(CCOPT) -I$(CONCERTINCDIR) -I$(CPLEXINCDIR)
    CCLNDIRS  = -L$(CPLEXLIBDIR) -L$(CONCERTLIBDIR)
    CCLNFLAGS = -lconcert -lilocplex -lcplex -lm -pthread -ldl

    OBJS2 += $(TARGETDIR)/MIPCertificate.o
endif

$(TARGETDIR)/bornAgain: $(OBJS2)
	$(CCC) $(CCFLAGS) $(CCLNDIRS) -o $(TARGETDIR)/bornAgain $(OBJS2) $(CCLNFLAGS)

$(TARGETDIR)/%.o: %.cpp
	$(CCC) -c $(CCFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	$(RM) $(OBJS2)
