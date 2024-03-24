import itertools
import pandas as pd

# function to judge if (all filter words in the string) 
# and (none exclusion word in the string) 
def filter_exclude(string_list:list[str], word_list:list[tuple], exclusion=[]):
    
    return_list = []
    for string in string_list:
        if all(True if i in string else False for i in word_list) and \
           all(True if j not in string else False for j in exclusion):
            return_list.append(string)
            
    return return_list

# function to filter GAEZ imgs and make a df
def get_img_df(img_path:list[str], exclusion:list[str] = ['None'], **kwargs):

    # get k,v
    keys = kwargs.keys()
    
    # formatting the vals and exclusion to list
    vals = [i if isinstance(i,list) else [i] for i in kwargs.values()]
    if not isinstance(exclusion, list):
        exclusion = [exclusion]

    # get the combinations from vals. E.g. [[A,B],[C]] ==> [[A,C],[B,C]]
    combination = list(itertools.product(*vals))

    # filter the img_path using combination
    filtered_list = []
    for word in combination:
        filtered = filter_exclude(img_path, word)     
        if len(filtered) == 0:
            print(f'{word} have no corresponding img!')
        elif len(filtered) == 1:
            filtered_list.append(word+(filtered[0],))
        else:
            filtered = filter_exclude(img_path,word,exclusion)
            if len(filtered) == 1:
                filtered_list.append(word+(filtered[0],))
            else:
                print(f'{word} have {len(filtered)} corresponding img!')
    
    # construct the df
    if len(filtered_list) > 0:
        df = pd.DataFrame(filtered_list)
        df.columns = list(keys) + ['path']
        return df
    else:
        print('\nNone img was filtered, find below to exclude the replicates\n')
        return filtered