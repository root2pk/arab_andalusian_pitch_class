# arab_andalusian_pitch_class
Companion repository for AMPLab assignment 2. Analysis of Arab Andalusian music scores to determine trends between pitch class histograms of scores across melodic and rhythmic frameworks.

## How to use the repository
`pre-process.py` is a script to pre-process the metadata and load Track objects associated with each score to `tracks.pkl`. The script has already been run and the files are available in the repository. The code is provided for reference.

`example_plot_pdfs` is an example notebook to compute the probability distribution functions of the notes in each score, and plot both unweighted (accounting for only note occurences) and weighted (accounting for note durations) pitch histograms.

`extract_non_weighted` is an example notebook to extract the nonweighted PDFs for each track, group them by tab and mizan, and plot graphs illustrating similarity metrics of scores with the same tab but different mizan.

`extract_weighted` is an example notebook that does the same as above but with weighted PDFs.

