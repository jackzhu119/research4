# Installed-source excerpts

Excerpts only; original installed sources were not modified. Line numbers are one-based and inclusive.

## lib/vhf/nr_direct.c

SHA-256: `9c181daa8ce689e312e548694f47a2935105094fab1e685e82b4bf485e0639c4`

Lines 233–265:

```text
  233  JKArray *CVHFallocate_JKArray(JKOperator *op, int *shls_slice, int *ao_loc,
  234                                int ncomp, int nblock, int size_limit)
  235  {
  236          JKArray *jkarray = malloc(sizeof(JKArray));
  237          int ibra = op->ibra_shl0;
  238          int iket = op->iket_shl0;
  239          int obra = op->obra_shl0;
  240          int oket = op->oket_shl0;
  241          int v_bra_sh0 = shls_slice[obra];
  242          int v_ket_sh0 = shls_slice[oket];
  243          jkarray->v_ket_nsh  = shls_slice[oket+1] - shls_slice[oket];
  244          jkarray->dm_dims[0] = ao_loc[shls_slice[ibra+1]] - ao_loc[shls_slice[ibra]];
  245          jkarray->dm_dims[1] = ao_loc[shls_slice[iket+1]] - ao_loc[shls_slice[iket]];
  246          jkarray->offset0_outptr = v_bra_sh0 * jkarray->v_ket_nsh + v_ket_sh0;
  247          jkarray->nblock = nblock;
  248          int *outptr = malloc(sizeof(int) * nblock * nblock);
  249          jkarray->outptr = outptr;
  250          int i;
  251          for (i = 0; i < nblock * nblock; i++) {
  252                  outptr[i] = NOVALUE;
  253          }
  254          jkarray->stack_size = 0;
  255          jkarray->data = malloc(sizeof(double) * (size_limit + 136*136));
  256          if (jkarray->data == NULL) {
  257                  fprintf(stderr, "malloc(%zu) failed in CVHFallocate_JKArray\n",
  258                          sizeof(double) * (size_limit + 136*136));
  259                  exit(1);
  260          }
  261          jkarray->ncomp = ncomp;
  262          int keys_max = size_limit / (AO_BLOCK_SIZE*AO_BLOCK_SIZE*ncomp);
  263          jkarray->keys_cache = malloc(sizeof(int) * keys_max);
  264          jkarray->key_counts = 0;
  265          return jkarray;
```

Lines 276–302:

```text
  276  double *CVHFallocate_and_reorder_dm(JKOperator *op, double *dm,
  277                                      int *shls_slice, int *ao_loc)
  278  {
  279          int ibra = op->ibra_shl0;
  280          int iket = op->iket_shl0;
  281          int ish0 = shls_slice[ibra];
  282          int jsh0 = shls_slice[iket];
  283          int ish1 = shls_slice[ibra+1];
  284          int jsh1 = shls_slice[iket+1];
  285          int ioff = ao_loc[ish0];
  286          int joff = ao_loc[jsh0];
  287          int nrow = ao_loc[ish1] - ioff;
  288          int ncol = ao_loc[jsh1] - joff;
  289          double *out = malloc(sizeof(double) * nrow*ncol);
  290          int ish, jsh, i0, i1, j0, j1, i, j, ij;
  291  
  292          ij = 0;
  293          for (ish = ish0; ish < ish1; ish++) {
  294          for (jsh = jsh0; jsh < jsh1; jsh++) {
  295                  i0 = ao_loc[ish  ] - ioff;
  296                  i1 = ao_loc[ish+1] - ioff;
  297                  j0 = ao_loc[jsh  ] - joff;
  298                  j1 = ao_loc[jsh+1] - joff;
  299                  for (i = i0; i < i1; i++) {
  300                  for (j = j0; j < j1; j++, ij++) {
  301                          out[ij] = dm[i*ncol+j];
  302                  } }
```

Lines 307–345:

```text
  307  void CVHFzero_out_vjk(double *vjk, JKOperator *op,
  308                        int *shls_slice, int *ao_loc, int ncomp)
  309  {
  310          int obra = op->obra_shl0;
  311          int oket = op->oket_shl0;
  312          int ish0 = shls_slice[obra];
  313          int jsh0 = shls_slice[oket];
  314          int ish1 = shls_slice[obra+1];
  315          int jsh1 = shls_slice[oket+1];
  316          int nbra = ao_loc[ish1] - ao_loc[ish0];
  317          int nket = ao_loc[jsh1] - ao_loc[jsh0];
  318          NPdset0(vjk, ((size_t)nbra) * nket * ncomp);
  319  }
  320  
  321  // Divide shls into subblocks with roughly equal number of AOs in each block
  322  int CVHFshls_block_partition(int *block_loc, int *shls_slice, int *ao_loc,
  323                               int block_size)
  324  {
  325          int ish0 = shls_slice[0];
  326          int ish1 = shls_slice[1];
  327          int count = 0;
  328          if (ish0 >= ish1) {
  329                  return count;
  330          }
  331  
  332          int ao_loc_last = ao_loc[ish0];
  333          int ish;
  334  
  335          count = 1;
  336          block_loc[0] = ish0;
  337          for (ish = ish0 + 1; ish < ish1; ish++) {
  338                  if (ao_loc[ish+1] - ao_loc_last > block_size) {
  339                          block_loc[count] = ish;
  340                          count++;
  341                          ao_loc_last = ao_loc[ish];
  342                  }
  343          }
  344          block_loc[count] = ish1;
  345          return count;
```

Lines 349–375:

```text
  349  /*
  350   * drv loop over ij, generate eris of kl for given ij, call fjk to
  351   * calculate vj, vk.
  352   * 
  353   * n_dm is the number of dms for one [array(ij|kl)], it is also the size of dms and vjk
  354   * ncomp is the number of components that produced by intor
  355   * shls_slice = [ishstart, ishend, jshstart, jshend, kshstart, kshend, lshstart, lshend]
  356   *
  357   * ao_loc[i+1] = ao_loc[i] + CINTcgto_spheric(i, bas)  for i = 0..nbas
  358   *
  359   * Return [(ptr[ncomp,nao,nao] in C-contiguous) for ptr in vjk]
  360   */
  361  void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
  362                         double **dms, double **vjk, int n_dm, int ncomp,
  363                         int *shls_slice, int *ao_loc,
  364                         CINTOpt *cintopt, CVHFOpt *vhfopt,
  365                         int *atm, int natm, int *bas, int nbas, double *env)
  366  {
  367          IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
  368                  cintopt, ncomp};
  369          int idm;
  370          double *tile_dms[n_dm];
  371          for (idm = 0; idm < n_dm; idm++) {
  372                  CVHFzero_out_vjk(vjk[idm], jkop[idm], shls_slice, ao_loc, ncomp);
  373                  tile_dms[idm] = CVHFallocate_and_reorder_dm(jkop[idm], dms[idm],
  374                                                              shls_slice, ao_loc);
  375          }
```

Lines 377–428:

```text
  377          int64_t di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
  378          int64_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
  379                                                 atm, natm, bas, nbas, env);
  380          int ish0 = shls_slice[0];
  381          int ish1 = shls_slice[1];
  382          int jsh0 = shls_slice[2];
  383          int jsh1 = shls_slice[3];
  384          int ksh0 = shls_slice[4];
  385          int ksh1 = shls_slice[5];
  386          int lsh0 = shls_slice[6];
  387          int lsh1 = shls_slice[7];
  388          int nish = ish1 - ish0;
  389          int njsh = jsh1 - jsh0;
  390          int nksh = ksh1 - ksh0;
  391          int nlsh = lsh1 - lsh0;
  392          int *block_iloc = malloc(sizeof(int) * (nish + njsh + nksh + nlsh + 4));
  393          int *block_jloc = block_iloc + nish + 1;
  394          int *block_kloc = block_jloc + njsh + 1;
  395          int *block_lloc = block_kloc + nksh + 1;
  396          // size_t to keep nblock^3 from overflowing for large molecules. The
  397          // same fix was applied to nr_sr_vhf.c.
  398          size_t nblock_i = CVHFshls_block_partition(block_iloc, shls_slice+0, ao_loc, AO_BLOCK_SIZE);
  399          size_t nblock_j = CVHFshls_block_partition(block_jloc, shls_slice+2, ao_loc, AO_BLOCK_SIZE);
  400          size_t nblock_k = CVHFshls_block_partition(block_kloc, shls_slice+4, ao_loc, AO_BLOCK_SIZE);
  401          size_t nblock_l = CVHFshls_block_partition(block_lloc, shls_slice+6, ao_loc, AO_BLOCK_SIZE);
  402          size_t nblock_kl = nblock_k * nblock_l;
  403          size_t nblock_jkl = nblock_j * nblock_kl;
  404          int nblock_max = MAX(nblock_i, nblock_j);
  405          nblock_max = MAX(nblock_max, nblock_k);
  406          nblock_max = MAX(nblock_max, nblock_l);
  407          // up to 3.2 GB per thread.
  408          int64_t size_limit = (400000000 - di*di*di*di*ncomp - cache_size) / n_dm;
  409          if (size_limit < 0) {
  410                  fprintf(stderr, "Insufficient memory for caching CVHFnr_direct_drv intermediates\n");
  411                  exit(1);
  412          }
  413  
  414  #pragma omp parallel
  415  {
  416          int ioff = ao_loc[ish0];
  417          int joff = ao_loc[jsh0];
  418          int koff = ao_loc[ksh0];
  419          int loff = ao_loc[lsh0];
  420          int i, j, k, l, n;
  421          size_t r, blk_id;
  422          JKArray *v_priv[n_dm];
  423          for (i = 0; i < n_dm; i++) {
  424                  v_priv[i] = CVHFallocate_JKArray(jkop[i], shls_slice, ao_loc,
  425                                                   ncomp, nblock_max, size_limit);
  426          }
  427          double *buf = malloc(sizeof(double) * (di*di*di*di*ncomp + di*di*2 + cache_size));
  428          double *cache = buf + di*di*di*di*ncomp;
```

Lines 429–483:

```text
  429  #pragma omp for nowait schedule(dynamic, 1)
  430          for (blk_id = 0; blk_id < nblock_jkl; blk_id++) {
  431                  // TODO: add schwarz ineq conditioner
  432                  r = blk_id;
  433                  j = r / nblock_kl ; r = r % nblock_kl;
  434                  k = r / nblock_l  ; r = r % nblock_l;
  435                  l = r;
  436                  int j0 = ao_loc[block_jloc[j]];
  437                  int k0 = ao_loc[block_kloc[k]];
  438                  int l0 = ao_loc[block_lloc[l]];
  439                  int j1 = ao_loc[block_jloc[j+1]];
  440                  int k1 = ao_loc[block_kloc[k+1]];
  441                  int l1 = ao_loc[block_lloc[l+1]];
  442                  for (n = 0; n < n_dm; n++) {
  443                          JKArray *pv = v_priv[n];
  444                          pv->ao_off[1] = j0 - joff;
  445                          pv->ao_off[2] = k0 - koff;
  446                          pv->ao_off[3] = l0 - loff;
  447                          pv->shape[1] = j1 - j0;
  448                          pv->shape[2] = k1 - k0;
  449                          pv->shape[3] = l1 - l0;
  450                          pv->block_quartets[1] = j;
  451                          pv->block_quartets[2] = k;
  452                          pv->block_quartets[3] = l;
  453                  }
  454                  for (i = 0; i < nblock_i; i++) {
  455                          int i0 = ao_loc[block_iloc[i]];
  456                          int i1 = ao_loc[block_iloc[i+1]];
  457                          for (n = 0; n < n_dm; n++) {
  458                                  JKArray *pv = v_priv[n];
  459                                  pv->ao_off[0] = i0 - ioff;
  460                                  pv->shape[0] = i1 - i0;
  461                                  pv->block_quartets[0] = i;
  462                          }
  463                          (*fdot)(intor, jkop, v_priv, tile_dms, buf, cache, n_dm,
  464                                  block_iloc+i, block_jloc+j, block_kloc+k, block_lloc+l,
  465                                  vhfopt, &envs);
  466                  }
  467                  for (n = 0; n < n_dm; n++) {
  468                          if (v_priv[n]->stack_size >= size_limit) {
  469  #pragma omp critical
  470          (*jkop[n]->write_back)(vjk[n], v_priv[n], shls_slice, ao_loc,
  471                                 block_iloc, block_jloc, block_kloc, block_lloc);
  472                          }
  473                  }
  474          }
  475          for (n = 0; n < n_dm; n++) {
  476                  if (v_priv[n]->stack_size > 0) {
  477  #pragma omp critical
  478          (*jkop[n]->write_back)(vjk[n], v_priv[n], shls_slice, ao_loc,
  479                                 block_iloc, block_jloc, block_kloc, block_lloc);
  480                  }
  481                  CVHFdeallocate_JKArray(v_priv[n]);
  482          }
  483          free(buf);
```

## lib/vhf/nr_direct_dot.c

SHA-256: `3db439ca3438b8291ac3510e463d8b8cb7c7ba66d8e95f305161843819745fc8`

Lines 38–44:

```text
   38  #define JKOP_DATA_SIZE(obra, oket) \
   39          static size_t JKOperator_data_size_##obra##oket(int *shls_slice, int *ao_loc) \
   40  { \
   41          int nbra = ao_loc[shls_slice[obra##SH1]] - ao_loc[shls_slice[obra##SH0]]; \
   42          int nket = ao_loc[shls_slice[oket##SH1]] - ao_loc[shls_slice[oket##SH0]]; \
   43          return nbra * nket; \
   44  }
```

Lines 58–114:

```text
   58  void JKOperator_write_back(double *vjk, JKArray *jkarray, int *ao_loc,
   59                             int *ishls, int *jshls, int *block_iloc, int *block_jloc)
   60  {
   61          int ish0 = ishls[0];
   62          int ish1 = ishls[1];
   63          int jsh0 = jshls[0];
   64          int jsh1 = jshls[1];
   65          size_t vrow = ao_loc[ish1] - ao_loc[ish0];
   66          size_t vcol = ao_loc[jsh1] - ao_loc[jsh0];
   67          int ncomp = jkarray->ncomp;
   68          int voffset = ao_loc[ish0] * vcol + ao_loc[jsh0];
   69          int nblock = jkarray->nblock;
   70          int key_counts = jkarray->key_counts;
   71          int *keys_cache = jkarray->keys_cache;
   72          int *offsets_dic = jkarray->outptr;
   73          double *jkarray_data = jkarray->data;
   74          int key_id, key, block_i, block_j;
   75          int i, j, ish, jsh, i0, j0;
   76          int di, dj, icomp;
   77          int block_i0, block_j0, block_dj;
   78          double *data, *pd, *pv;
   79  
   80          for (key_id = 0; key_id < key_counts; key_id++) {
   81                  key = keys_cache[key_id];
   82                  block_i = key / nblock;
   83                  block_j = key % nblock;
   84                  ish0 = block_iloc[block_i];
   85                  ish1 = block_iloc[block_i+1];
   86                  jsh0 = block_jloc[block_j];
   87                  jsh1 = block_jloc[block_j+1];
   88                  block_i0 = ao_loc[ish0];
   89                  block_j0 = ao_loc[jsh0];
   90                  block_dj = ao_loc[jsh1] - ao_loc[jsh0];
   91  
   92                  data = jkarray_data + offsets_dic[key];
   93                  offsets_dic[key] = NOVALUE;
   94                  for (ish = ish0; ish < ish1; ish++) {
   95                  for (jsh = jsh0; jsh < jsh1; jsh++) {
   96                          i0 = ao_loc[ish];
   97                          j0 = ao_loc[jsh];
   98                          di = ao_loc[ish+1] - i0;
   99                          dj = ao_loc[jsh+1] - j0;
  100                          pd = data + ((i0 - block_i0) * block_dj + (j0 - block_j0) * di) * ncomp;
  101                          pv = vjk + i0*vcol+j0 - voffset;
  102                          for (icomp = 0; icomp < ncomp; icomp++) {
  103                                  for (i = 0; i < di; i++) {
  104                                  for (j = 0; j < dj; j++) {
  105                                          pv[i*vcol+j] += pd[i*dj+j];
  106                                  } }
  107                                  pv += vrow * vcol;
  108                                  pd += di * dj;
  109                          }
  110                  } }
  111          }
  112          jkarray->stack_size = 0;
  113          jkarray->key_counts = 0;
  114  }
```

Lines 180–207:

```text
  180  #define LOCATE(v, i, j) \
  181          int d##i##j = d##i * d##j; \
  182          _key = block[i##SH]*out->nblock+block[j##SH]; \
  183          _poutptr = out->outptr + _key; \
  184          if (*_poutptr == NOVALUE) { \
  185                  *_poutptr = out->stack_size; \
  186                  int v_size = shape[i##SH]*shape[j##SH]; \
  187                  out->stack_size += v_size * ncomp; \
  188                  NPdset0(out->data+*_poutptr, v_size*ncomp); \
  189                  out->keys_cache[out->key_counts] = _key; \
  190                  out->key_counts++; \
  191          } \
  192          double *v = out->data + *_poutptr; \
  193          v += ((i##0-ao_off[i##SH]) * shape[j##SH] + (j##0-ao_off[j##SH]) * d##i) * ncomp;
  194  
  195  #define DECLARE(v, i, j) \
  196          int ncomp = out->ncomp; \
  197          int ncol = out->dm_dims[1]; \
  198          int di = i1 - i0; \
  199          int dj = j1 - j0; \
  200          int dk = k1 - k0; \
  201          int dl = l1 - l0; \
  202          int _key; \
  203          int *_poutptr; \
  204          int *shape = out->shape; \
  205          int *block = out->block_quartets; \
  206          int *ao_off = out->ao_off; \
  207          LOCATE(v, i, j)
```

## lib/vhf/nr_direct.h

SHA-256: `df7f96a170594e1d180a44cf2f0b1b5d1b00bbadd01e3ef288c23ecc15149a8f`

Lines 22–42:

```text
   22  #define AO_BLOCK_SIZE   64
   23  
   24  #define NOVALUE 0x7fffffff
   25  
   26  #if !defined(HAVE_DEFINED_INTORENV_H)
   27  #define HAVE_DEFINED_INTORENV_H
   28  typedef struct {
   29          int v_ket_nsh;  /* v_ket_sh1 - v_ket_sh0 */
   30          int offset0_outptr;  /* v_bra_sh0 * v_ket_nsh + v_ket_sh0 */
   31          int dm_dims[2];
   32          int *outptr;   /* Offset array to index the data which are stored in stack */
   33          double *data;  /* Stack to store data */
   34          int stack_size;  /* How many data have been used */
   35          int ncomp;
   36          int nblock;
   37          int ao_off[4];  /* Offsets for the first AO in the block */
   38          int shape[4];  /* The shape of eri for the target block */
   39          int block_quartets[4]; /* 4 block Ids */
   40          int *keys_cache;  /* key (=i*nblock+j) to track allocated data */
   41          int key_counts; /* Number of allocated data block */
   42  } JKArray;
```

Lines 44–58:

```text
   44  typedef struct {
   45          int ibra_shl0;  // = 0, 2, 4, 6. The index in shls_slice
   46          int iket_shl0;
   47          int obra_shl0;
   48          int oket_shl0;
   49          void (*contract)(double *eri, double *dm, JKArray *vjk, int *shls,
   50                           int i0, int i1, int j0, int j1,
   51                           int k0, int k1, int l0, int l1);
   52          size_t (*data_size)(int *shls_slice, int *ao_loc);
   53          void (*sanity_check)(int *shls_slice);
   54          void (*write_back)(double *vjk, JKArray *jkarray,
   55                             int *shls_slice, int *ao_loc,
   56                             int *block_iloc, int *block_jloc,
   57                             int *block_kloc, int *block_lloc);
   58  } JKOperator;
```

## lib/np_helper/np_helper.c

SHA-256: `2c875766b8b721287ef8e210cae6ca94bcea2ea101930a7a31f1aeb80a656dea`

Lines 20–26:

```text
   20  void NPdset0(double *p, const size_t n)
   21  {
   22          size_t i;
   23          for (i = 0; i < n; i++) {
   24                  p[i] = 0;
   25          }
   26  }
```

## grad/rhf.py

SHA-256: `d30fd1ab2a00997183802da28da775f6a29a6c9705c55b19c04d66a1a575267f`

Lines 206–219:

```text
  206  def get_j(mol, dm):
  207      '''
  208      J = ((-nabla i) j| kl) D_lk
  209      '''
  210      vhfopt = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFgrad_j_prescreen',
  211                            dmcondname='CVHFnr_dm_cond1')
  212      vhfopt.q_cond = _calc_q_cond(mol, vhfopt)
  213      intor = mol._add_suffix('int2e_ip1')
  214      vj = _vhf.direct_mapdm(intor,  # (nabla i,j|k,l)
  215                             's2kl', # ip1_sph has k>=l,
  216                             'lk->s1ij',
  217                             dm, 3, # xyz, 3 components
  218                             mol._atm, mol._bas, mol._env, vhfopt=vhfopt)
  219      return -vj
```

## grad/rks.py

SHA-256: `cffc3e9efa10cf1b547d757e9d165c1000377e9159c794f87464a79c8c93ab46`

Lines 75–82:

```text
   75                  max_memory=max_memory, verbose=ks_grad.verbose)
   76              vxc += vnlc
   77      t0 = logger.timer(ks_grad, 'vxc', *t0)
   78  
   79      if not ni.libxc.is_hybrid_xc(mf.xc):
   80          vj = ks_grad.get_j(mol, dm)
   81          vxc += vj
   82      else:
```

## scf/_vhf.py

SHA-256: `ef0a04f43338780a7f57d9268f9634c1079ae989ba3030d6154c8612798f82ba`

Lines 431–465:

```text
  431  # call all fjk for each dm. The return has the shape
  432  # [len(jkdescript),len(dms),ncomp,nao,nao]
  433  # jkdescript: 'ij->s1kl', 'kl->s2ij', ...
  434  def direct_mapdm(intor, aosym, jkdescript,
  435                   dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
  436                   shls_slice=None, shls_excludes=None, out=None,
  437                   optimize_sr=False):
  438      if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
  439          dms = dms[numpy.newaxis,:,:]
  440          single_dm = True
  441      else:
  442          single_dm = False
  443      intor = ascint3(intor)
  444      if vhfopt is None:
  445          cvhfopt = lib.c_null_ptr()
  446          cintopt = None
  447      else:
  448          vhfopt.set_dm(dms, atm, bas, env)
  449          cvhfopt = vhfopt._this
  450          cintopt = vhfopt._cintopt
  451  
  452      n_dm = len(dms)
  453      dms = [numpy.asarray(dm, order='C', dtype=numpy.double) for dm in dms]
  454      if isinstance(jkdescript, str):
  455          jkscripts = (jkdescript,)
  456      else:
  457          jkscripts = jkdescript
  458      n_jk = len(jkscripts)
  459  
  460      # make n_jk copies of dms
  461      dms = dms * n_jk
  462      # make n_dm copies for each jk script
  463      jkscripts = numpy.repeat(jkscripts, n_dm)
  464  
  465      vjk = nr_direct_drv(intor, aosym, jkscripts, dms, ncomp, atm, bas, env,
```
