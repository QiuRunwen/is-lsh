Summarized by Prof. Wenbin Zhu (i@zhuwb.com)
date: 2022 May 17

Files:
    twonorm.tar.gz      Downloaded on 17 May 2022 from https://www.cs.toronto.edu/~delve/data/twonorm/desc.html
       Source/gen       The gawk code for generating data, where x_{ij} is rounded to 4th decimial places
       Dataset.data.gz  Generated data
    phpxijhaP.arff      Downloaded on 17 May 2022 from https://www.openml.org/d/1507
                        The uploader claim the data is originated from https://www.cs.toronto.edu/~delve/data/twonorm/desc.html
                        However compare to ringnorm.tar.gz#Dataset.data.gz, the class label is changed:
                            class 0 -> 1
                            class 1 -> 2
    twonorm.csv         Extracted data from phpxijhaP.arff and changed class labels as follows by Miss Ying Fu
                            ringnorm.tar.gz   phpWfYmlu.arff   twonorm.csv
                                class 0             1                1
                                class 1             2               -1

    	
