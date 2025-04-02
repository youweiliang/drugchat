The data format that stores data in json files is not scalable. We recommend storing data in the following format to avoid data duplication and make the training scalable with large data. This effectively avoids OOM when the data is very large (e.g., millions of entries). 

Specifically, a data directory should contains the following 4 files (which are compressed pandas DataFrame files). The compound_idx, question_idx, and label in "QA.parquet" can be looked up in the remaining 3 files. They can be read by `pandas.read_parquet` in python.

"QA.parquet" (for general data) or "QA_2.parquet" (for bioassays)
```
compound_idx  question_idx  label
       11876          4938      1
       11877          4938      1
      226398          4977      1
```

"compound_smiles.parquet"
```
 compound_idx                                            smiles
            0              COc1cc(C(=O)NN=Cc2ccnc3ccccc23)ccc1O
            1                 O=C(NNCc1cccc2ccccc12)c1ccc(O)cc1
            2        COc1cc(C(=O)N/N=C/c2ccc(C(C)(C)C)cc2)ccc1O
```

"question_idx.parquet"
```
 question_idx                                                           question
            0                         Is this compound cytotoxic to HepG2 cells?
            1 Is this compound cytotoxic to human skeletal muscle cells (HSkMC)?
            2    Is this compound cytotoxic to human foetal lung (IMR-90) cells?
```

"label2text.parquet"
```
label text
    0   No
    1  Yes
```

If you are using the previously not scalable data format: `{SMILES String: [ [Question1 , Answer1], [Question2 , Answer2]... ] }`, you can use this script `dataset/convert_json2pd.py` to convert the not scalable data format to the scalable data format. 

Then, specify the directory in the build_info of the datasets in your train config yaml file to start training.
