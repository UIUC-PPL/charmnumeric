#include <charmtyles/core/server.hpp>

#include "backend.hpp"

CkGroupID create_dag_group() { return CProxy_ArrayDAGGroup::ckNew(); }

class Main : public CBase_Main {
  public:
    Main(CkArgMsg* m) {
        create_dag_group();
        // Server::initialize is called from ArrayDAGGroup::proxies_ready
        // after all PEs have received partition proxies via reduction.
    }

    Main(CkMigrateMessage* m) : CBase_Main(m) { create_dag_group(); }

    void pup(PUP::er& p) {}
};

#include "charmtyles.def.h"
#include "charmnumeric.def.h"
