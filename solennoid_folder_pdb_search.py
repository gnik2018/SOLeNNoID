from solenoid_detector_onehot_v3 import *
import pandas as pd
import numpy as np
from Bio.PDB.PDBIO import PDBIO
import argparse
import os


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='SOLeNNoID script for solenoid residue prediction. Please provide a directory using the -d flag.')
parser.add_argument("-d", "--Directory", help = "Directory with .pdb files")
parser.add_argument("-f", "--File_list", help = "File list with .pdb files")

args = parser.parse_args()
directory=args.Directory
file_list=np.loadtxt(args.File_list,dtype='str').tolist()


#load model and set class dictionary
loaded_model=load_model('/Users/georginikov/SOLeNNoID/trained_models/Conv2D_dmca_multiclass_norm_filesplit_s128_b64_e100_split0.2_1645094358_diagonal_checkpoint.h5')
class_dict={0:'non',1:'beta',2:'alphabeta',3:'alpha'}

def scandir_recursive(directory):
    for entry in os.scandir(directory):
        if entry.is_dir(follow_symlinks=False):
            yield from scandir_recursive(entry.path)
        else:
            yield entry.path

def get_data(structure_path,structure_id):
    if structure_path.endswith('cif'):
        parser = MMCIFParser()
    elif structure_path.endswith('pdb'):
        parser=PDBParser()

    #load structure and chain ids
    structure=parser.get_structure("sds",structure_path)
    chain_ids=[x.get_id() for x in structure[0]]

    scores=[]
    dfs=np.array([])

    for chain in chain_ids:

        try:
            #set up protein and prediction classes
            protein1=Protein(structure_path,chain=chain, structure=structure)
            protein1.set_up()
            first_residue,distance_matrix,chain_list=protein1.return_matrix_and_first_res()
            maxlength=protein1.indices[-1]-first_residue
            prediction1=Prediction(first_res_chain=first_residue, distance_matrix_chain=distance_matrix,loaded_model=loaded_model,sub_matrix_size=128, maxlength=maxlength)
            final_scores=prediction1.get_final_onehot()

            #get argmaxed predictions
            scores_argmaxed=np.argmax(final_scores,axis=1)

            #get residue indices
            indices=[i+first_residue for i in range(len(final_scores))]
            #get chain column

            chain_column=[chain]*len(final_scores)
            #put it all together in a dictionary and make a pandas DataFrame
            beta=[indices[i] for i in range(len(scores_argmaxed)) if scores_argmaxed[i]==1]
            alphabeta=[indices[i] for i in range(len(scores_argmaxed)) if scores_argmaxed[i]==2]
            alpha=[indices[i] for i in range(len(scores_argmaxed)) if scores_argmaxed[i]==3]
            #save numbers of residues of each solenoid type
            beta_res = len(beta)
            alphabeta_res=len(alphabeta)
            alpha_res=len(alpha)
            total_res=maxlength
            beta_proc=100*(beta_res/total_res)
            alphabeta_proc=100*(alphabeta_res/total_res)
            alpha_proc=100*(alpha_res/total_res)

            line=[structure_id,chain,total_res,beta,beta_res,beta_proc,alphabeta,alphabeta_res,alphabeta_proc,alpha,alpha_res,alpha_proc]
            dfs=np.append(dfs,line)
        except:
            pass



    return dfs.reshape(int(len(dfs)/12),12)


#file_list=list(scandir_recursive(directory))
print(len(file_list))
full_dfs=[]
duds=[]
count=0
count2=0
for file in file_list:
    if file.endswith('cif'):
        structure_path=file
        structure_id=file.split('/')[-1][:4]
        print(structure_id)
        try:
            local_df=get_data(structure_path,structure_id)
            full_dfs.append(local_df)
        except:
            pass
    '''
    if count < 1000:
        if file.endswith('cif'):
            structure_path=file
            structure_id=file.split('/')[-1][:4]
            print(structure_id)
            try:
                local_df=get_data(structure_path,structure_id)
                full_dfs.append(local_df)
            except:
                pass
            count+=1
    elif count == 1000:
        if file.endswith('cif'):
            structure_path=file
            structure_id=file.split('/')[-1][:4]
            print(structure_id)
            try:
                local_df=get_data(structure_path,structure_id)
                full_dfs.append(local_df)
            except:
                pass
        count2+=1
        concat=np.concatenate(full_dfs,axis=0)
        df=pd.DataFrame(concat, columns=['structure_id','chain_id','total_num','beta','beta_num','beta_percentage','alphabeta','alphabeta_num','alphabeta_percentage','alpha','alpha_num','alpha_percentage'])
        df.to_csv(f'final_chunk_{count2}_pdb.csv')
        del df
        count=0
        full_dfs=[]
        '''
concat=np.concatenate(full_dfs,axis=0)
df=pd.DataFrame(concat, columns=['structure_id','chain_id','total_num','beta','beta_num','beta_percentage','alphabeta','alphabeta_num','alphabeta_percentage','alpha','alpha_num','alpha_percentage'])
df.to_csv(f'final_chunk_24_pdb.csv')

#df.to_csv(f'final_chunk_{count2}_pdb.csv')
