# Open the input file for reading

def rank_prediction(type):
    with open(f'result/per_molecule_pred_of_{type}_set.res', 'r') as f:
        # Read in the scores and true labels from each line and store them in a list of tuples
        data = [(float(line.split('\t')[0]), line.split('\t')[1] ) for line in
                f.readlines()]

    # Rank the scores in ascending order
    ranked_data = sorted(data, key=lambda x: x[0], reverse=True)

    # Open the output file for writing
    with open(f'result/ranked_mol_score_{type}.res', 'w') as f:
        # Write out the ranked scores and true labels to the output file
        for i, (score, label) in enumerate(ranked_data):
            f.write(f"{i}\t{score}\t{label}")

