
import pandas as pd
import numpy as np
import random
import math
from itertools import chain
from data_preprocessing import DrugDataset




def split_data(all_pos_tups, all_drugs, c_f=1, k=1):

     num_drugs = len(all_drugs)
     train_tups = []
     s1_tups = []
     s2_tups = []

     test_ratio = 0.2
     drug_index = list(range(num_drugs))
     np.random.shuffle(drug_index)
     test_index = drug_index[0:int(num_drugs*test_ratio)]
     train_index = drug_index[int(num_drugs*test_ratio):]
     all_drugs = np.array(list(all_drugs))
     train_drugs = all_drugs[train_index]
     test_drugs = all_drugs[test_index]
     for h, t, r, ind in all_pos_tups:
        if (h in train_drugs) and (t in train_drugs):
            train_tups.append((h, t, r, ind))
        elif (h in test_drugs) and (t in test_drugs):
            s1_tups.append((h, t, r, ind))
        else:
            s2_tups.append((h, t, r, ind))

     print(f"# train_tups: {len(train_tups)}")
     print(f"# s1_tups: {len(s1_tups)}")
     print(f"# s2_tups: {len(s2_tups)}")
     return train_tups, s1_tups, s2_tups

# reserved_drugs = set()
    # guaranteed_rels = set()
    # reserved_tups = []
    # random.shuffle(all_pos_tups)
    # remaining_tups = []
    # for h, t, r, ind in all_pos_tups:
    #     if r not in guaranteed_rels:
    #         reserved_tups.append((h, t, r, ind))
    #         guaranteed_rels.add(r)
    #         reserved_drugs.add(h)
    #         reserved_drugs.add(t)
    #     else:
    #         remaining_tups.append((h, t, r, ind))
    # print(f"# of all drugs: {len(all_drugs)}")
    # print(f"# of drugs to guarantee all relations are present: {len(reserved_drugs)}")
    #
    # remaining_drugs = list(all_drugs - reserved_drugs)
    # n_test = math.ceil(len(remaining_drugs) / c_f)
    #
    # for i in range(k):
    #     print(f"{i+1}/{k}")
    #     c_start = i * n_test
    #     c_end = c_start + n_test
    #     candidate_cold_state_drugs = set(remaining_drugs[c_start:c_end])
    #     existing_drugs = set(list(reserved_drugs) + remaining_drugs[0:c_start] + remaining_drugs[c_end:])
    #     print(f"# candidate new drugs: {len(candidate_cold_state_drugs)}")
    #     print(f"# existing drugs: {len(existing_drugs)}")
    #
    #     assert len(all_drugs) == len(candidate_cold_state_drugs) + len(
    #         existing_drugs), 'Error # of drugs should be equal to the sum of new drugs and existing drugs'
    #
    #     train_tups = list(reserved_tups)
    #     s1_tups = []
    #     s2_tups = []
    #     for h, t, r, ind in remaining_tups:
    #         if (h in existing_drugs) and (t in existing_drugs):
    #             train_tups.append((h, t, r, ind))
    #         elif (h in candidate_cold_state_drugs) and (t in candidate_cold_state_drugs):
    #             s1_tups.append((h, t, r, ind))
    #         else:
    #             s2_tups.append((h, t, r, ind))
    #     print(f"# train_tups: {len(train_tups)}")
    #     print(f"# s1_tups: {len(s1_tups)}")
    #     print(f"# s2_tups: {len(s2_tups)}")
    #     assert len(all_pos_tups) == len(train_tups) + len(s1_tups) + len(s2_tups)
    #     train_drugs = set(chain.from_iterable((h, t) for h, t, *_ in train_tups))
    #     s1_drugs = set(chain.from_iterable((h, t) for h, t, *_ in s1_tups))
    #     s2_drugs = set(chain.from_iterable((h, t) for h, t, *_ in s2_tups))
    #
    #     # assert len(s1_drugs - existing_drugs) == len(s1_drugs) // 2
    #     # assert len(s1_drugs - candidate_cold_state_drugs) == len(s1_drugs) // 2
    #     assert (s1_drugs | (s2_drugs - existing_drugs)) == candidate_cold_state_drugs
    #     assert (train_drugs | (s2_drugs - candidate_cold_state_drugs)) == existing_drugs
    #     assert (s1_drugs - train_drugs) == s1_drugs

        # train_data = DrugDataset(train_tups, disjoint_split=True)
        # s2_data = DrugDataset(s2_tups, disjoint_split=False)
        # s1_data = DrugDataset(s1_tups, disjoint_split=True)
        #
        # assert set(s2_data.drug_ids) == all_drugs
        # assert len(set(s1_data.drug_ids) & set(train_data.drug_ids)) == 0

        # yield train_tups, s1_tups, s2_tups
        # return train_tups, s1_tups, s2_tups
        # yield train_data, s1_data, s2_data
