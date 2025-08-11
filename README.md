# Can Media Act as a Soft Regulator of Safe AI Development? A Game Theoretical Analysis


Official repository for the paper: 

**Can Media Act as a Soft Regulator of Safe AI Development? A Game Theoretical Analysis**  
Henrique Correia da Fonseca<sup>1,*</sup>, António Fernandes<sup>1</sup>, Zhao Song<sup>2</sup>, *et al.*  
Published in *[Journal / Conference]*, 2025  
[DOI link] | [arXiv link]



<details>
<summary>Full Author List</summary>

**Authors**
1. Henrique Correia da Fonseca<sup>1,∗</sup>
2. António Fernandes<sup>1</sup>
3. Zhao Song<sup>2</sup>
4. Theodor Cimpeanu<sup>3</sup>
5. Nataliya Balabanova<sup>4</sup>
6. Adeela Bashir<sup>2</sup>
7. Paolo Bova<sup>2</sup>
8. Alessio Buscemi<sup>5</sup>
9. Alessandro Di Stefano<sup>2</sup>
10. Manh Hong Duong<sup>4</sup>
11. Elias Fernandez Domingos<sup>6,7</sup>
12. Ndidi Bianca Ogbo<sup>2</sup>
13. Simon T. Powers<sup>8</sup>,
14. Daniele Proverbio<sup>9</sup>
15. Zia Ush Shamszaman<sup>2</sup>
16. Fernando P. Santos<sup>10</sup>
17. The Anh Han<sup>2</sup>
18. Marcus Krellner

**Affiliations**

<sup>1</sup> INESC-ID and Instituto Superior Técnico, Universidade de Lisboa

<sup>2</sup> School Computing, Engineering and Digital Technologies, Teesside University

<sup>3</sup> School of Mathematics and Statistics, University of St Andrews

<sup>4</sup> School of Mathematics, University of Birmingham

<sup>5</sup> Luxembourg Institute of Science and Technology

<sup>6</sup> Machine Learning Group, Université libre de Bruxelles

<sup>7</sup> AI Lab, Vrije Universiteit Brussel

<sup>8</sup> Division of Computing Science and Mathematics, University of Stirling

<sup>9</sup> Department of Industrial Engineering, University of Trento

<sup>10</sup> University of Amsterdam

<sup>*</sup> corresponding author: henrique.c.fonseca@tecnico.ulisboa.pt

</details>


### Abstract
When developers of artificial intelligence (AI) products need to decide between profit and safety for the users, they likely choose profit. Untrustworthy AI technology must come packaged with tangible negative consequences. Here, we envisage those consequences as the loss of reputation caused by media coverage of their misdeeds, disseminated to the public. We explore whether media coverage has the potential to push AI creators into the production of safe products, enabling widespread adoption of AI technology. We created artificial populations of self-interested creators and users and studied them through the lens of evolutionary game theory. Our results reveal that media is indeed able to foster cooperation between creators and users, but not always. Cooperation does not evolve if the quality of the information provided by the media is not reliable enough, or if the costs of either accessing media or ensuring safety are too high. By shaping public perception and holding developers accountable, media emerges as a powerful soft regulator -- guiding AI safety even in the absence of formal government oversight.


# Repository Structure

```
inputs/                    # input files     
    └── input.yaml         # example file
outputs/                   # output files
    ├── heatmaps/          # generated heatmaps
    └── time_series/       # generated time series
.gitignore                 # gitignore
creator.py                 # creator class
main.py                    # main
plotting.py                # plotting functions
LICENSE                    # MIT license
README.md                  # this file
requirements.txt           # pip requirements
simulator.py               # simulator class
user.py                    # user class
```

### Requirements

- Python 3.10+

# Usage

1. Clone the repository

    ``` bash
    git clone https://github.com/cfonsecahenrique/media-AI-governance.git

    cd media-AI-governance
    ```

2. Install dependencies

    ``` bash
    pip install -r requirements.txt
    ```

    **Note:** you may want first to setup a virtual environment to keep your project's dependencies isolated, preventing version conflicts and ensuring reproducible results.

3. Run experiments

    1. Define the running, simulation, game parameters (and heatmap conditions if applicable) in `<input file>.yaml`. The file must be located inside `inputs` directory.

        For better understanding of the input file, please check the variables description below:

        <details>

        <summary>Variables description</summary>

        - `running`:

            - `runs` - number of simulations
            - `cores` - number of cores to run parallel simulations (use `all` to get the maximum number of available cores in your CPU)
        
        - `simulation`:

            - `type` - simulation type (`time_series` or `heatmap`)
            - `user population size` - size of user population
            - `creator population size` - size of creator population
            - `user selection strength` - intensity of selection in users decision making process
            - `creator selection strength` - intensity of selection in creators decision making process
            - `user mutation probability` - mutation probability (percentual) in user's evolutionary step
            - `creator mutation probability` - mutation probability (percentual) in creator's evolutionary step
            - `generations` - number of generations computed in each simulation
            - `convergence period` - percentage of generations to consider convergent
            - `user initialization` - initial strategy of users (`-1` for random initialization; `0`/`1`/`2`/`3` for all users with `ALL_REJECT`/`ALL_ADOPT`/`BAD_MEDIA`/`GOOD_MEDIA` respectively)
            - `creator initialization` - initial strategy of creators (`-1` for random initialization; `0`/`1` for `DEFECT`/`COOPERATE` respectively)

        - `parameters`:

            - `media quality` - probability that the recommendation of a commentator is correct
            - `user benefit` - benefit a user receives when adopting a safe technology
            - `user cost` - cost a user incurs when adopting an unsafe technology
            - `cost investigation` - cost of informed recommendation
            - `creator benefit` - benefit a creator receives when (for user or commentator) their technology is adopted
            - `creator cost` - additional cost of creating safe AI (the cost of creating unsafe AI is normalised to 0)
           

        - `heatmap`:

            - `vars` - heatmap (x, y) variables
            - `v1_start` - start point for variable x
            - `v1_end` - final point for variable x
            - `v1_steps` - incrementing step for variable x
            - `v1_scale` - scale for variable x
            - `v2_start` - start point for variable y
            - `v2_end` - final point for variable y
            - `v2_steps` - incrementing step for variable y
            - `v2_scale` - scale for variable y
        
        </details> 

    2. Run the experiment as 
    ```bash
    python main.py <input file>.yaml
    ```

    3. Results of th experiment will be stored inside the `outputs` directory.

# Citation

If you use this code or data in your research, please cite as:

```
@article{
}
```

# License

This project is licensed under the [MIT License](./LICENSE).

# Contact

For questions or collaborations, please contact henrique.c.fonseca@tecnico.ulisboa.pt or open an issue.