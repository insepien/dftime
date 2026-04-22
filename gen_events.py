from src import *
import scipy.stats as st

def main(args):
    a0_pop_min=10
    a0_pop_max=20

    # 1. Define a large batch size (e.g., 10,000)
    batch_size = 10000 
    reso = 0.1
    np.random.seed(0)
    # 2. Generate arrays for all candidates at once
    a = np.random.uniform(a0_pop_min, a0_pop_max, batch_size)
    q = np.random.uniform(1, 90, batch_size)
    z = 0.2#np.random.uniform(0.18, 0.2, batch_size)
    pobs = psel(a, z, q, reso) #* 1/10*1/99
    u = np.random.uniform(0, pobs.max(), batch_size)

    # 3. Apply the mask
    mask = u < pobs

    # 4. Filter
    accepted_a = a[mask]
    accepted_q = q[mask]
    accepted_pobs = pobs[mask]

    Nobs = 100
    # generate more than the observe size to remove the ones with negative values
    a0_samp = accepted_a[:Nobs]
    q_samp = accepted_q[:Nobs]
    samples_aq = np.stack([a0_samp,q_samp],axis=1)

    def gen_event_posteriors(true_param,smin,smax,n_post_samps = 101):
        # random sample a measurement about true param, then random sample measurement posterior
        single_event_posteriors = [st.norm(st.norm(a, s).rvs(), s).rvs(n_post_samps) 
                            for a, s in zip(true_param, np.random.uniform(smin,smax, size = len(true_param)))]
        # find posterior with negative value
        find_neg = lambda arr: np.array([i for i in range(len(arr)) if np.min(arr[i])<0])
        neg_ind = find_neg(single_event_posteriors)
        # print(neg_ind)
        # niter = 0
        while len(neg_ind) != 0:
            for i in neg_ind:
                s = np.random.uniform(smin,smax)
                a = true_param[i]
                single_event_posteriors[i] = st.norm(st.norm(a, s).rvs(), s).rvs(n_post_samps) 
            neg_ind = find_neg(single_event_posteriors)
            # niter+=1
        return single_event_posteriors

    single_event_posteriors_a = gen_event_posteriors(a0_samp,1,3)
    single_event_posteriors_q = gen_event_posteriors(q_samp,0.1,1)
    single_event_posteriors_aq = np.stack([single_event_posteriors_a,single_event_posteriors_q],axis=2)

    for i in range(Nobs):
        filepath = os.path.join(args.outdir,f"event_{i}.txt")
        np.savetxt(filepath,single_event_posteriors_aq[i,:,:])

if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        generate event posteriors on a,q
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=str, default="/home/insepien/dftime/data/events", help="output directory")
    args = parser.parse_args()
    main(args)