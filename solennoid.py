from solenoid_detector_onehot_v3 import *
import pandas as pd
import numpy as np
from Bio.PDB.PDBIO import PDBIO
import argparse


# Initialize parser
parser = argparse.ArgumentParser(description='SOLeNNoID script for solenoid residue prediction. Please provide a structure using the -i flag. Predictions are output either in a .csv file, or a .pdb file with b factors replaced with solenoid class numerical values. 1 - beta-solenoid, 2 - alpha/beta-solenoid, 3 - alpha-solenoid.')
# Adding optional argument
parser.add_argument("-i", "--Input", help = "Input file")
parser.add_argument("-csv","--Csv",choices=['Y','N'],help='Output csv with predictions')
parser.add_argument('-pdb','--Pdb',choices=['Y','N'],help='Output PDB file with predictions as b factors')
args = parser.parse_args()
structure_path=args.Input

def output_predictions(structure_path):
    """
    Takes the path to the protein structure path as the 'structure_path'
    argument, checks if the structure is a .pdb or .mmcif structure and
    loads the correct BioPython parser, then the structure is processed,
    the SOLeNNoID model is loaded and each chain is processed to one-hot
    predictions. The predictions for each chain are then concatenated for
    final output.
    """
    #load appropriate parser for the filetype
    if structure_path.endswith('cif'):
        parser = MMCIFParser()
    elif structure_path.endswith('pdb'):
        parser=PDBParser()

    #load structure and chain ids
    structure=parser.get_structure("sds",structure_path)
    chain_ids=[x.get_id() for x in structure[0]]

    scores=[]
    dfs=[]

    #load model and set class dictionary
    loaded_model=load_model('./trained_models/Conv2D_dmca_multiclass_norm_filesplit_s128_b64_e100_split0.2_1645094358_diagonal_checkpoint.h5')
    class_dict={0:'non',1:'beta',2:'alphabeta',3:'alpha'}

    for chain in chain_ids:

        try:

            #set up protein and prediction classes
            protein1=Protein(structure_path,chain=chain, structure=structure)
            protein1.set_up()
            first_residue,distance_matrix,chain_list=protein1.return_matrix_and_first_res()
            maxlength=protein1.indices[-1]-first_residue
            prediction1=Prediction(first_res_chain=first_residue, distance_matrix_chain=distance_matrix,loaded_model=loaded_model,sub_matrix_size=128, maxlength=maxlength)
            final_scores=prediction1.get_final_onehot()

            scores.append(final_scores)
            #get argmaxed predictions
            scores_argmaxed=np.argmax(final_scores,axis=1)
            #convert from numerical to word value
            scores_translated=[class_dict[i] for i in scores_argmaxed]
            #get residue names
            names=[structure[0][chain].get_unpacked_list()[i].get_resname() for i in range(len(final_scores))]
            #get residue indices
            indices=[i+first_residue for i in range(len(final_scores))]
            #get chain column
            chain_column=[chain]*len(final_scores)
            #put it all together in a dictionary and make a pandas DataFrame
            pd_dict={'chain':chain_column, 'index':indices,'residue':names, 'one_hot':final_scores,'class':scores_translated}
            df=pd.DataFrame.from_dict(pd_dict)
            dfs.append(df)

            #Update structure b factors to have prediction numerical values
            for i in range(len(final_scores)):
                for atom in structure[0][chain].get_unpacked_list()[i]:
                    atom=atom.set_bfactor(scores_argmaxed[i])
        except:
            pass

    #output section
    scores_concat=np.concatenate(scores,axis=0)
    total_df=pd.concat(dfs)
    io=PDBIO()
    io.set_structure(structure)

    if args.Csv=='Y':
        total_df.to_csv(f"{structure_path[0:-4]}_predictions.csv")

    if args.Pdb=='Y':
        io.save(f"{structure_path[0:-4]}_predictions.pdb")
        print('USE THIS PYMOL SCRIPT TO COLOUR BY SOLENOID CLASS:')
        print('spectrum b, grey_magenta_cyan_orange, minimum=0,maximum=3')

    return

output_predictions(structure_path)
