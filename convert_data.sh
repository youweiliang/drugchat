

# conda activate rdkit

python dataset/smiles2image.py --smiles_path data_public/ChEMBL_QA_train.json --save_dir data_public/ChEMBL/train
python dataset/smiles2graph.py --smiles_path data_public/ChEMBL_QA_train.json --save_dir data_public/ChEMBL/train
python dataset/smiles2image.py --smiles_path data_public/ChEMBL_QA_val.json --save_dir data_public/ChEMBL/val
python dataset/smiles2graph.py --smiles_path data_public/ChEMBL_QA_val.json --save_dir data_public/ChEMBL/val
python dataset/smiles2image.py --smiles_path data_public/ChEMBL_QA_test.json --save_dir data_public/ChEMBL/test
python dataset/smiles2graph.py --smiles_path data_public/ChEMBL_QA_test.json --save_dir data_public/ChEMBL/test

python dataset/smiles2image.py --smiles_path data_public/PubChem_QA.json --save_dir data_public/PubChem/train
python dataset/smiles2graph.py --smiles_path data_public/PubChem_QA.json --save_dir data_public/PubChem/train

python dataset/smiles2image.py --smiles_path data_public/DrugBank_train.json --save_dir data_public/DrugBank/train
python dataset/smiles2graph.py --smiles_path data_public/DrugBank_train.json --save_dir data_public/DrugBank/train
python dataset/smiles2image.py --smiles_path data_public/DrugBank_val.json --save_dir data_public/DrugBank/val
python dataset/smiles2graph.py --smiles_path data_public/DrugBank_val.json --save_dir data_public/DrugBank/val
python dataset/smiles2image.py --smiles_path data_public/DrugBank_test.json --save_dir data_public/DrugBank/test
python dataset/smiles2graph.py --smiles_path data_public/DrugBank_test.json --save_dir data_public/DrugBank/test