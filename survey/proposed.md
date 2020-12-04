# Proposed Survey Methodology

Survey: Anonymous Link https://mit.co1.qualtrics.com/jfe/form/SV_6X4UtmVnYRSPk45

* 1 dataset, 3 tasks, within-subject survey
* Approximate time 20 minutes
* Proposed compensation: $3.25 (based on Prolific rate suggestions: https://www.prolific.co/pricing)


## Analysis

### Task Relevance
* Goal: Evaluate whether survey tasks are representative of real-world data analysis tasks.

#### Question

> The general task (i.e. ignoring dataset domain) is reflective of tasks you, as a > participant, might perform in your own data analyses.

* Ideal: Tasks skew higher on Likert scale, reflecting real-world relevance of task
* Presented *once* per task (no treatment/control)
* Report:
  - Mean and SD of 1 - 7 Likert choices for each task
  - Histogram of Likert choices for each task


### Retrieval Quality and System
* Goal: Evaluate utility of survey snippets to data analysis
* Goal: Evaluate our system's ranking of code snippets versus a random ranking


* Setup:
  - Two snippet groups: treatment (ours) and control (random retrieval and ordering)
  - Users see each group for a task, before they move on to the next task
  - Order of groups presented is randomized for each task

#### Question: Number of fragments relevant

> Which fragment snippets do you consider relevant to the task?

* Ideal: Our system has a larger number of relevant fragments than the control
* Report:
  - Mean and SD of number marked as relevant for each group/task
  - Histogram for number marked relevant for each group/task
  - Test: Wilcoxon-Signed ranked test
    - Nonparametric
    - Ok with non-continuous values (i.e. counts)
    - Bonferroni correction for number of tasks (3)


#### Question: Ordering of fragments

> Which snippet do you consider to be most relevant to the task?
> (choose closest option, if none are relevant)

* Ideal: Relevant fragments are ranked higher by our system than by the control
* Report:
  - Mean and SD of rank for chosen fragment for each group/task
  - Histogram of rank for chosen fragment each group/task
  - Test: Wilcoxon-Signed ranked test
    - Nonparametric
    - Ok with non-continuous values (i.e. counts)
    - Bonferroni correction for number of tasks (3)


#### Question: System is helpful
> Access to these snippets makes completing the task easier


* Note: (José) I've kept the phrase as "these snippets" rather than "relevant snippets" as we'd like to compare all snippets produced by the two systems, not just the relevant snippets produced by the two systems


* Ideal: Our system scores higher on Likert than the control
* Report:
  - Mean and SD of 1 - 7 Likert choices for each group/task
  - Histogram of Likert choices for each group/task
  - Test: Wilcoxon-Signed ranked test
    - Nonparametric
    - Ok with non-continuous values (i.e. counts)
    - Bonferroni correction for number of tasks (3)
