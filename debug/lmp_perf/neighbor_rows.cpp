// SPDX-License-Identifier: LGPL-3.0-or-later
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "force.h"
#include "input.h"
#include "kokkos.h"
#include "library.h"
#include "lammps.h"
#include "memory.h"
#include "nbin_kokkos.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list_kokkos.h"
#include "npair.h"
#include "pair_zero.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace LAMMPS_NS;
static int probe_flags = 0;

class NeighborProbe : public PairZero {
 public:
  explicit NeighborProbe(LAMMPS *lmp) : PairZero(lmp) {
    kokkosable = 1;
    execution_space = Device;
    datamask_read = datamask_modify = EMPTY_MASK;
  }
  ~NeighborProbe() override {
    if (cutghost) memory->destroy(cutghost);
  }
  void compute(int eflag, int vflag) override { ev_init(eflag,vflag); }
  void init_style() override {
    auto *request = neighbor->add_request(this,probe_flags);
    request->set_kokkos_device(1);
    if (probe_flags & NeighConst::REQ_GHOST) {
      ghostneigh = 1;
      const int count = atom->ntypes;
      if (!cutghost) memory->create(cutghost,count+1,count+1,"probe:cutghost");
      for (int i=1;i<=count;i++)
        for (int j=1;j<=count;j++) cutghost[i][j]=cut_global;
    }
  }
};

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  if (argc < 6) return 2;
  const std::string mode=argv[3];
  if (mode=="full" || mode=="ghost") probe_flags|=NeighConst::REQ_FULL;
  if (mode=="ghost") probe_flags|=NeighConst::REQ_GHOST;
  if (mode=="size") probe_flags|=NeighConst::REQ_SIZE | NeighConst::REQ_HISTORY;
  Force::pair_styles().set_plugin("neighbor_probe/kk",[](LAMMPS *lmp)->Pair* { return new NeighborProbe(lmp); });
  LAMMPS::argv args={"neighbor_rows","-log","log.lammps","-screen","none","-k","on","g","1","-sf","kk"};
  for (int i=6;i<argc;i++) args.emplace_back(argv[i]);
  {
    auto lmp=std::make_unique<LAMMPS>(args,MPI_COMM_WORLD);
    lmp->input->file(argv[1]);
    auto *list=dynamic_cast<NeighListKokkos<LMPDeviceType>*>(lmp->force->pair->list);
    if (!list || !list->np) return 3;
    auto *bins=dynamic_cast<NBinKokkos<LMPDeviceType>*>(list->np->nb);
    if (!bins) return 4;
    auto *atoms=static_cast<AtomKokkos*>(lmp->atom);
    const uint64_t fields=X_MASK | TYPE_MASK | MASK_MASK | TAG_MASK | RADIUS_MASK;
    atoms->sync(Host,fields);
    auto hx=atoms->k_x.view_host();
    auto ht=atoms->k_type.view_host();
    auto hm=atoms->k_mask.view_host();
    auto hg=atoms->k_tag.view_host();
    auto hr=atoms->k_radius.view_host();
    struct Record { double x[3]; double radius; tagint tag; int type; int mask; };
    const bool read=std::string(argv[5])=="read";
    FILE *snapshot=std::fopen(argv[4],read?"rb":"wb");
    if (!snapshot) return 5;
    int dimensions[2]={atoms->nlocal,atoms->nghost};
    if (read) {
      int saved[2];
      if (std::fread(saved,sizeof(int),2,snapshot)!=2 ||
          saved[0]!=dimensions[0] || saved[1]!=dimensions[1]) return 6;
    } else std::fwrite(dimensions,sizeof(int),2,snapshot);
    for (int i=0;i<atoms->nlocal+atoms->nghost;i++) {
      Record record{};
      if (read) {
        if (std::fread(&record,sizeof(record),1,snapshot)!=1) return 7;
        for (int d=0;d<3;d++) hx(i,d)=record.x[d];
        ht(i)=record.type; hm(i)=record.mask; hg(i)=record.tag;
        if (hr.data()) hr(i)=record.radius;
      } else {
        for (int d=0;d<3;d++) record.x[d]=hx(i,d);
        record.type=ht(i); record.mask=hm(i); record.tag=hg(i);
        record.radius=hr.data()?hr(i):0;
        std::fwrite(&record,sizeof(record),1,snapshot);
      }
    }
    std::fclose(snapshot);
    if (read) {
      atoms->modified(Host,fields);
      atoms->sync(Device,fields);
    }
    bins->bin_atoms();
    list->np->build_setup();
    bins->k_bincount.sync_host();
    bins->k_bins.sync_host();
    auto counts=bins->k_bincount.view_host();
    auto contents=bins->k_bins.view_host();
    FILE *out=std::fopen(argv[2],"w");
    std::fprintf(out,"local %d ghost %d cap %d bins %zu\n",lmp->atom->nlocal,lmp->atom->nghost,bins->atoms_per_bin,counts.extent(0));
    for (size_t b=0;b<counts.extent(0);b++) {
      std::vector<int> row;
      for (int j=0;j<counts(b);j++) row.push_back(contents(b,j));
      std::sort(row.begin(),row.end());
      std::fprintf(out,"bin %zu",b);
      for (size_t j=0;j<row.size();j++) {
        contents(b,j)=row[j];
        std::fprintf(out," %d",row[j]);
      }
      std::fprintf(out,"\n");
    }
    bins->k_bins.modify_host();
    bins->k_bins.sync_device();
    list->np->build(list);
    auto num=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),list->d_numneigh);
    auto ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),list->d_ilist);
    auto neighbors=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),list->d_neighbors);
    std::fprintf(out,"rows %d %d\n",list->inum,list->gnum);
    for (int ii=0;ii<list->inum+list->gnum;ii++) {
      const int i=ids(ii);
      std::fprintf(out,"row %d",i);
      for (int j=0;j<num(i);j++) std::fprintf(out," %u",static_cast<unsigned>(neighbors(i,j)));
      std::fprintf(out,"\n");
    }
    std::fclose(out);
  }
  lammps_kokkos_finalize();
  MPI_Finalize();
}
