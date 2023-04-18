# to be added after the HamObj initialization:

    numstates   = 50000
    rbs         = cupy.zeros((numstates, nchain+1), dtype="uint16")
    if False:
        rbs[:5,2]   = cupy.arange(5)
        rbs[5:10,1] = cupy.arange(5,10)
        rbs[10:,3]  = cupy.arange(1, 11)
        rbs[:10,0]  = 1
        rbs[10:15,0]= 2
        rbs[15:,0]  = 0
    else:
        rbs[:,0]    = cupy.arange(numstates) % nchain
#        rbs[:,0]    = 4
#        rbs[:,1:]   = (cupy.random.rand(100) * 100).astype(int).reshape((20, nchain))
        rbs[:,-2] = cupy.random.rand(numstates) * 100
        rbs[:,-5] = cupy.random.rand(numstates) * 100
        rbs[:,-1] = 42
    print(rbs)

    bs          = HamObj.compress_states(rbs)
#    print("initial id: %i." % id(bs))
    bs_orig     = bs.copy()
#    print(bs)
#    print(HamObj.decompress_all_sites(bs))
    print(numpy.all(HamObj.decompress_all_sites(bs) == rbs))
    HamObj.add_phonon_at_exc(bs)
#    print("id after adding phonons: %i." % id(bs))
    print("\nPhonons at exc were added.")
#    print(bs)
#    print(HamObj.decompress_all_sites(bs))
    print((HamObj.decompress_all_sites(bs) - rbs)[:20])
    diffmask    = numpy.any((HamObj.decompress_all_sites(bs) - rbs)[:,1:] != cupy.tile(cupy.eye(nchain), (int(numpy.ceil(numstates/nchain)), 1))[:numstates], axis=1)
    print(diffmask.sum() == 0)
#    print(HamObj.decompress_all_sites(bs_orig)[diffmask])
#    print(HamObj.decompress_all_sites(bs)[diffmask])
    HamObj.rem_phonon_at_exc(bs)
    print("\nPhonons at exc were removed.")
#    print(bs)
#    print(HamObj.decompress_all_sites(bs))
#    print(HamObj.decompress_all_sites(bs) - rbs)
    print(numpy.all(bs == bs_orig))
    print(HamObj.get_phonon_at_exc(bs))
#    print("final id: %i." % id(bs))
#    print(HamObj.sum_all_phonons(bs, omega=numpy.arange(nchain)*1.42))
#    print(HamObj.posbitwidth + HamObj.QHObitwidth.cumsum() - HamObj.QHObitwidth)
