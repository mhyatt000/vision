class SeriesExperiment(Experiment):
    """5 Seeds of 2-fold cross validation"""
    """ sometimes """

    def __init__(self, cfg):
        self.versions = os.path.join(cfg.out, "versions.json")
        pass

    def mk_versions(self, force=False):
        "allows different versions of the experiment to be saved to different folders"

        if not cfg.master:
            return
        mkdir(cfg.out)
        if os.path.exists(self.versions) and not force:  
            # dont remake if you can start in the middle
            # unless you are forced to by partition
            return

        exp = []
        ordered = lambda x: x.reverse() if cfg.EXP.REVERSE else x

        LO_combos = [None]
        if cfg.LOADER.LEAVE_N_OUT:
            classes = [
                [i for i in range(cfg.LOADER.NCLASSES)] for _ in range(cfg.LOADER.LEAVE_N_OUT)
            ]
            LO_combos = list(itertools.product(*classes))
            LO_combos = [sorted(set(x)) for x in LO_combos]
            temp = []
            for LO in LO_combos:
                if len(LO) == cfg.LOADER.LEAVE_N_OUT and not LO in temp:
                    temp.append(LO)
            LO_combos = temp

        for LO in LO_combos:
            for seed, swap in itertools.product(ordered([0, 1, 2, 3, 4]), ordered([False, True])):
                e = {"LO": LO, "seed": seed, "swap": swap} if cfg.EXP.MULTISEED else {"LO":LO}
                exp.append(e)
                mkdir(os.path.join(cfg.OUT, out.d2s(e)))

        # TODO: partition experiments on the nodes
        if cfg.EXP.PARTITION:
            assert len(cfg.nodes) == len(exp), f'{len(cfg.nodes)} != {len(exp)} ... you cannot partition here'
            exp = {k:v for k,v in zip(cfg.nodes,exp)}

        with open(self.versions, "w") as file:
            json.dump(exp, file)

    def pop_versions(self):
        """pop the first experiment version off the file"""
        if cfg.master:  # only rank 0 pops
            with open(self.versions, "r") as file:
                exp = json.load(file)

            if cfg.EXP.PARTITION:
                exp = {k:v for k,v in exp.items() if k != cfg.nodename}
            else:
                exp = exp[1:]

            with open(self.versions, "w") as file:
                json.dump(exp, file)
        else: 
            time.sleep(2) # so that you read the right version next time

    def destroy(self):
        """destroys objects to allocate data for next experiment in series"""

        keys = list(self.__dict__.keys())
        for k in keys:
            if k != 'versions':
                delattr(self, k)


    # TODO: can you generalize for many iterations of any hparam? ie: LO
    def run(self):
        self.mk_versions()
        dist.barrier()

        while True:
            version = out.get_exp_version()
            if not version:
                break
            cfg.SEED = int(version["seed"])
            setup_seed(cfg.SEED)

            super().__init__()
            super().run()
            self.pop_versions()
            self.destroy()
            dist.barrier()

