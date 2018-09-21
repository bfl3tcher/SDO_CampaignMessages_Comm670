
# coding: utf-8

# # Data Processing
# 
# **Script Goals**
# * Drop unnecessary columns
# * Combine variables from test conditions
# * Split up multiple dataframes based on test conditions
# * Create nets, feature engineering variables
# * Create mean variables for SDO, etc.

# ## Step 1 - Preparing Dataset for Analysis

# **SDO Exploration - Data Munging**
# **Coding - 1-8 Likert scale, No Response coded as 0**
# 
# **Pro Social Dominance**
# * SDO_Q5_1 -- SDO1_Pro_TraitDominance
#     * An ideal society requires some groups to be on top and others to be on the bottom.
# * SDO_Q5_13 -- SDO2_Pro_TraitDominance
#     * Some groups of people are simply inferior to other groups.
# * SDO_Q5_7 -- SDO3-Pro_TraitAntiegalitarianism
#     * It is unjust to try to make groups equal.
# * SDO_Q5_3 -- SDO4-Pro_TraitAntiegalitarianism
#     * Group equality should not be our primary goal.
# 
# **Anti-Social Dominance**
# * SDO_Q5_6 -- SDO5-Con_TraitDominance (Reverse coded)
#     * Groups at the bottom are just as deserving as groups at the top.
# * SDO_Q5_2 -- SDO6-Con_TraitDominance (Reverse coded)
#     * No one group should dominate in society.
# * SDO_Q5_14 -- SDO7-Con_TraitAntiegalitariansim - (Reverse coded)
#     * We should do what we can to equalize conditions for different groups.    
# * SDO_Q5_4 -- SDO8-Con_TraitAntiegalitariansim (Reverse coded)
#     * We should work to give all groups an equal chance to succeed.
# 

# **Demographic Variables**
# * Age - Q11
# * Ethnicity - Q12
# * Sex - Q13

# **Political Variables - Exploration**
# * Political Ideology 
#     * **Political_Views_Q6_2** -- How would you describe your views on social issues?
#     * **Political_Views_Q6_3** -- Overall, how would you describe your political ideology?
#     * **Political_Views_Q6_4** -- How would you describe your views on economic issues?
#     
# * Public Trust
#     * **Public_Trust_Q7_13** -- I think most public officials can be trusted.
#     * **Public_Trust_Q7_6** -- I don't think public officials care much what people like me think.
#     * **Public_Trust_Q7_2** -- People like me don't have a say about what the government does.
# 
# * Political Interest
#     * **Q8_13** -- Generally speaking, how interested are you in politics and elections?
# * Voter Liklihood
#     * **Q9_13** -- How likely is it that you will vote in the next presidential election?

# In[1]:


# importing libraries
import numpy as np
import pandas as pd


# **Importing Data**

# In[2]:


df = pd.read_csv('SDO_Campaigns_HumanReadable.csv')

# removing unnecessary columns
df.drop(
    columns=[
        'StartDate',
        'EndDate',
        'Status',
        'Progress',
        'Duration__in_seconds_',
        'Finished',
        'DistributionChannel',
        'RecordedDate',
        'Q3_Consent'
    ], 
    axis=0,
    inplace=True
)


# renaming SDO variables
df.rename(
    index=str, 
    columns={
        # Pro Trait dominance variables
        'SDO_Q5_1':'sdo1_Pro_Trait_Dom1',
        'SDO_Q5_13':'sdo13_Pro_Trait_Dom2',
        
        # Con Trait Dominance variables
        'SDO_Q5_6':'sdo6_Con_Trait_Dom2',                
        'SDO_Q5_2':'sdo2_Con_Trait_Dom1',
        
        # Pro Trait AntiEgalitarianism
        'SDO_Q5_7':'sdo7_Pro_Trait_AntiEgal1',        
        'SDO_Q5_3':'sdo3_Pro_Trait_AntiEgal2',

        # Con Trait AntiEgalitarianism
        'SDO_Q5_14':'sdo14_Con_Trait_AntiEgal1',        
        'SDO_Q5_4':'sdo4_Con_Trait_AntiEgal2'
    }, inplace=True)

# renaming Political Variables
df.rename(
    index=str, 
    columns={
        
        # Political Ideology
        'Political_Views_Q6_2':'ideol2_social',
        'Political_Views_Q6_3':'ideol3_self',
        'Political_Views_Q6_4':'ideol4_econ',
        
        # Public Trust
        'Public_Trust_Q7_13':'trust13_officials',        
        'Public_Trust_Q7_6':'trust6_nocare',
        'Public_Trust_Q7_2':'trust2_nosay',
        
        # Interest, Voter
        'Q8_13':'pol_interest',
        'Q9_13':'pol_vote',        
        
    }, inplace=True)


# renaming demo variables
df.rename(
    index=str, 
    columns={
        
        # demographic variables
        'Q11':'Age',
        'Q12':'Ethnicity',
        'Q13':'Sex'
    }, inplace=True)


# renaming Q20/Q21 - (H-SDO) & (Civil-Positive) -
df.rename(
    index=str, 
    columns={
        # Q20 -  
        'Q20_14':'Q20_mess14_imprtnt',
        'Q20_15':'Q20_mess15_inform',
        'Q20_13':'Q20_mess13_fair',
        
        # Q21 - 
        'Q21_1':'Q21_cand1_strong',
        'Q21_13':'Q21_cand13_relate',    
        'Q21_6':'Q21_cand6_weak',    
        'Q21_2':'Q21_cand2_dishonest',    
        'Q21_7':'Q21_cand7_friends',    
        'Q21_3':'Q21_cand3_aggressive',    
        'Q21_4':'Q21_cand4_moral',    
        'Q21_14':'Q21_cand14_competent',  
        'Q21_15':'Q21_cand15_votefor',  
        'Q21_16':'Q21_cand16_volunteer',          
        'Q21_17':'Q21_cand17_persuade',                  
    }, inplace=True)


# renaming Q30/Q31 - (L-SDO) & (Civil-Positive)
df.rename(
    index=str, 
    columns={
        # Q30 -  
        'Q30_14':'Q30_mess14_imprtnt',
        'Q30_15':'Q30_mess15_inform',
        'Q30_13':'Q30_mess13_fair',
        
        # Q31 - 
        'Q31_1':'Q31_cand1_strong',
        'Q31_13':'Q31_cand13_relate',    
        'Q31_6':'Q31_cand6_weak',    
        'Q31_2':'Q31_cand2_dishonest',    
        'Q31_7':'Q31_cand7_friends',    
        'Q31_3':'Q31_cand3_aggressive',    
        'Q31_4':'Q31_cand4_moral',    
        'Q31_14':'Q31_cand14_competent',  
        'Q31_15':'Q31_cand15_votefor',  
        'Q31_16':'Q31_cand16_volunteer',          
        'Q31_17':'Q31_cand17_persuade',                  
    }, inplace=True)

# renaming Q40/Q41 - (H-SDO) & (Civil-Negative) - 
df.rename(
    index=str, 
    columns={
        # Q40 -  
        'Q40_14':'Q40_mess14_imprtnt',
        'Q40_15':'Q40_mess15_inform',
        'Q40_13':'Q40_mess13_fair',
        
        # Q41 - 
        'Q41_1':'Q41_cand1_strong',
        'Q41_13':'Q41_cand13_relate',    
        'Q41_6':'Q41_cand6_weak',    
        'Q41_2':'Q41_cand2_dishonest',    
        'Q41_7':'Q41_cand7_friends',    
        'Q41_3':'Q41_cand3_aggressive',    
        'Q41_4':'Q41_cand4_moral',    
        'Q41_14':'Q41_cand14_competent',  
        'Q41_15':'Q41_cand15_votefor',  
        'Q41_16':'Q41_cand16_volunteer',          
        'Q41_17':'Q41_cand17_persuade',                  
    }, inplace=True)

# renaming Q50/Q51 - (L-SDO) & (Civil-Negative)
df.rename(
    index=str, 
    columns={
        # Q50 -  
        'Q50_14':'Q50_mess14_imprtnt',
        'Q50_15':'Q50_mess15_inform',
        'Q50_13':'Q50_mess13_fair',
        
        # Q51 - 
        'Q51_1':'Q51_cand1_strong',
        'Q51_13':'Q51_cand13_relate',    
        'Q51_6':'Q51_cand6_weak',    
        'Q51_2':'Q51_cand2_dishonest',    
        'Q51_7':'Q51_cand7_friends',    
        'Q51_3':'Q51_cand3_aggressive',    
        'Q51_4':'Q51_cand4_moral',    
        'Q51_14':'Q51_cand14_competent',  
        'Q51_15':'Q51_cand15_votefor',  
        'Q51_16':'Q51_cand16_volunteer',          
        'Q51_17':'Q51_cand17_persuade',                  
    }, inplace=True)


# renaming Q60/Q61 - (H-SDO) & (UnCivil) - 
df.rename(
    index=str, 
    columns={
        # Q60 -  
        'Q60_14':'Q60_mess14_imprtnt',
        'Q60_15':'Q60_mess15_inform',
        'Q60_13':'Q60_mess13_fair',
        
        # Q61 - 
        'Q61_1':'Q61_cand1_strong',
        'Q61_13':'Q61_cand13_relate',    
        'Q61_6':'Q61_cand6_weak',    
        'Q61_2':'Q61_cand2_dishonest',    
        'Q61_7':'Q61_cand7_friends',    
        'Q61_3':'Q61_cand3_aggressive',    
        'Q61_4':'Q61_cand4_moral',    
        'Q61_14':'Q61_cand14_competent',  
        'Q61_15':'Q61_cand15_votefor',  
        'Q61_16':'Q61_cand16_volunteer',          
        'Q61_17':'Q61_cand17_persuade',                  
    }, inplace=True)


# renaming Q70/Q71 - (L-SDO) & (UnCivil) -
df.rename(
    index=str, 
    columns={
        # Q70 -  
        'Q70_14':'Q70_mess14_imprtnt',
        'Q70_15':'Q70_mess15_inform',
        'Q70_13':'Q70_mess13_fair',
        
        # Q71 - 
        'Q71_1':'Q71_cand1_strong',
        'Q71_13':'Q71_cand13_relate',    
        'Q71_6':'Q71_cand6_weak',    
        'Q71_2':'Q71_cand2_dishonest',    
        'Q71_7':'Q71_cand7_friends',    
        'Q71_3':'Q71_cand3_aggressive',    
        'Q71_4':'Q71_cand4_moral',    
        'Q71_14':'Q71_cand14_competent',  
        'Q71_15':'Q71_cand15_votefor',  
        'Q71_16':'Q71_cand16_volunteer',          
        'Q71_17':'Q71_cand17_persuade',                  
    }, inplace=True)



df['pol_interest'] = df.pol_interest.str.replace('\(|\)', '')
df['pol_interest'] = df.pol_interest.str.replace('Highest Interest |Lowest Interest ', '')
df['pol_vote'] = df.pol_interest.str.replace('\(|\)', '')


# In[3]:


# Positive coding SDO (LowSDO)1-7(HighSDO)
SDO_values = {
    'NO RESPONSE': None,
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Slightly Disagree': 3,
    'Neither Agree nor Disagree':4,
    'Slightly Agree':5,
    'Agree':6,    
    'Strongly Agree':7
}

# reverse coded SDO (HighSDO)1-7(LowSDO)
SDO_ReverseCode = {
    'NO RESPONSE': None,
    'Strongly Disagree': 7,
    'Disagree': 6,
    'Slightly Disagree': 5,
    'Neither Agree nor Disagree':4,
    'Slightly Agree':3,
    'Agree':2,    
    'Strongly Agree':1
}

# Political Ideology (Very Liberal)1-7(Very Conservative)
ideology_values = {
    'NO RESPONSE': None,
    'Very Liberal': 1,
    'Liberal': 2,
    'Slightly Liberal': 3,
    'Neither Liberal nor Conservative':4,
    'Slightly Conservative':5,
    'Conservative':6,    
    'Very Conservative':7
}


# political trust (Strongly Disagree)1-7(Strongly Agree)
trust_values = {
    'NO RESPONSE': None,
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Slightly Disagree': 3,
    'Neither Agree nor Disagree':4,
    'Slightly Agree':5,
    'Agree':6,    
    'Strongly Agree':7
}


# Voter interest (LowInterest)1-4(High Interest)
interest_values = {
    'NO RESPONSE': None,
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Slightly Disagree': 3,
    'Neither Agree nor Disagree':4,
}


# vote turnout (NOT turnout)1-4(WILL turnout)
pol_voter_values = {
    'NO RESPONSE': None,
    '1': 1,
    '-2': 2,
    '-3': 3,
    '4':4,
}


# vote turnout (NOT turnout)1-4(WILL turnout)
pol_interest_values = {
    'NO RESPONSE': None,
    'Will definitely NOT vote 1': 1,
    '-2': 2,
    '-3': 3,
    'Will definitely vote 4':4,
}


# Message Values (Strongly Disagree)1-4(Strongly Agree)
mess_values = {
    'NO RESPONSE': None,
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Agree': 3,
    'Strongly Agree': 4   
}

# Candidate Values (Strongly Disagree)1-7(Strongly Agree)
cand_values = {
    'NO RESPONSE': None,
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Slightly Disagree': 3,
    'Neither Agree nor Disagree': 4,   
    'Slightly Agree': 5,
    'Agree': 6,
    'Strongly Agree': 7,    
}


# Candidate Values (Strongly Agree)1-7(Strongly Disagree)
cand_reverse_values = {
    'NO RESPONSE': None,
    'Strongly Agree': 1,    
    'Agree': 2,
    'Slightly Agree': 3,
    'Neither Agree nor Disagree': 4,   
    'Slightly Disagree': 5,
    'Disagree': 6,
    'Strongly Disagree': 7,
}


# In[4]:


# pro-SDO
df.sdo1_Pro_Trait_Dom1.replace(SDO_values, inplace=True)
df.sdo13_Pro_Trait_Dom2.replace(SDO_values, inplace=True)
df.sdo7_Pro_Trait_AntiEgal1.replace(SDO_values, inplace=True)
df.sdo3_Pro_Trait_AntiEgal2.replace(SDO_values, inplace=True)

# con-SDO
df.sdo2_Con_Trait_Dom1.replace(SDO_ReverseCode, inplace=True)
df.sdo6_Con_Trait_Dom2.replace(SDO_ReverseCode, inplace=True)
df.sdo14_Con_Trait_AntiEgal1.replace(SDO_ReverseCode, inplace=True)
df.sdo4_Con_Trait_AntiEgal2.replace(SDO_ReverseCode, inplace=True)

# political ideology
df.ideol3_self.replace(ideology_values, inplace=True)
df.ideol4_econ.replace(ideology_values, inplace=True)
df.ideol2_social.replace(ideology_values, inplace=True)

# political interest
df.pol_interest.replace(pol_interest_values, inplace=True)
df.pol_vote.replace(pol_voter_values, inplace=True)

# political trust
df.trust13_officials.replace(trust_values, inplace=True)
df.trust2_nosay.replace(trust_values, inplace=True)
df.trust6_nocare.replace(trust_values, inplace=True)


# Q20
df.Q20_mess13_fair.replace(mess_values, inplace=True)
df.Q20_mess14_imprtnt.replace(mess_values, inplace=True)
df.Q20_mess15_inform.replace(mess_values, inplace=True)

# Q21
df.Q21_cand1_strong.replace(cand_values, inplace=True)
df.Q21_cand2_dishonest.replace(cand_reverse_values, inplace=True)
df.Q21_cand3_aggressive.replace(cand_reverse_values, inplace=True)
df.Q21_cand4_moral.replace(cand_values, inplace=True)
df.Q21_cand6_weak.replace(cand_reverse_values, inplace=True)
df.Q21_cand7_friends.replace(cand_values, inplace=True)
df.Q21_cand13_relate.replace(cand_values, inplace=True)
df.Q21_cand14_competent.replace(cand_values, inplace=True)
df.Q21_cand15_votefor.replace(cand_values, inplace=True)
df.Q21_cand16_volunteer.replace(cand_values, inplace=True)
df.Q21_cand17_persuade.replace(cand_values, inplace=True)


# Q30
df.Q30_mess13_fair.replace(mess_values, inplace=True)
df.Q30_mess14_imprtnt.replace(mess_values, inplace=True)
df.Q30_mess15_inform.replace(mess_values, inplace=True)

# Q31
df.Q31_cand1_strong.replace(cand_values, inplace=True)
df.Q31_cand2_dishonest.replace(cand_reverse_values, inplace=True)
df.Q31_cand3_aggressive.replace(cand_reverse_values, inplace=True)
df.Q31_cand4_moral.replace(cand_values, inplace=True)
df.Q31_cand6_weak.replace(cand_reverse_values, inplace=True)
df.Q31_cand7_friends.replace(cand_values, inplace=True)
df.Q31_cand13_relate.replace(cand_values, inplace=True)
df.Q31_cand14_competent.replace(cand_values, inplace=True)
df.Q31_cand15_votefor.replace(cand_values, inplace=True)
df.Q31_cand16_volunteer.replace(cand_values, inplace=True)
df.Q31_cand17_persuade.replace(cand_values, inplace=True)


# Q40
df.Q40_mess13_fair.replace(mess_values, inplace=True)
df.Q40_mess14_imprtnt.replace(mess_values, inplace=True)
df.Q40_mess15_inform.replace(mess_values, inplace=True)

# Q41
df.Q41_cand1_strong.replace(cand_values, inplace=True)
df.Q41_cand2_dishonest.replace(cand_reverse_values, inplace=True)
df.Q41_cand3_aggressive.replace(cand_reverse_values, inplace=True)
df.Q41_cand4_moral.replace(cand_values, inplace=True)
df.Q41_cand6_weak.replace(cand_reverse_values, inplace=True)
df.Q41_cand7_friends.replace(cand_values, inplace=True)
df.Q41_cand13_relate.replace(cand_values, inplace=True)
df.Q41_cand14_competent.replace(cand_values, inplace=True)
df.Q41_cand15_votefor.replace(cand_values, inplace=True)
df.Q41_cand16_volunteer.replace(cand_values, inplace=True)
df.Q41_cand17_persuade.replace(cand_values, inplace=True)

# Q50
df.Q50_mess13_fair.replace(mess_values, inplace=True)
df.Q50_mess14_imprtnt.replace(mess_values, inplace=True)
df.Q50_mess15_inform.replace(mess_values, inplace=True)

# Q51
df.Q51_cand1_strong.replace(cand_values, inplace=True)
df.Q51_cand2_dishonest.replace(cand_reverse_values, inplace=True)
df.Q51_cand3_aggressive.replace(cand_reverse_values, inplace=True)
df.Q51_cand4_moral.replace(cand_values, inplace=True)
df.Q51_cand6_weak.replace(cand_reverse_values, inplace=True)
df.Q51_cand7_friends.replace(cand_values, inplace=True)
df.Q51_cand13_relate.replace(cand_values, inplace=True)
df.Q51_cand14_competent.replace(cand_values, inplace=True)
df.Q51_cand15_votefor.replace(cand_values, inplace=True)
df.Q51_cand16_volunteer.replace(cand_values, inplace=True)
df.Q51_cand17_persuade.replace(cand_values, inplace=True)


# Q60
df.Q60_mess13_fair.replace(mess_values, inplace=True)
df.Q60_mess14_imprtnt.replace(mess_values, inplace=True)
df.Q60_mess15_inform.replace(mess_values, inplace=True)

# Q61
df.Q61_cand1_strong.replace(cand_values, inplace=True)
df.Q61_cand2_dishonest.replace(cand_reverse_values, inplace=True)
df.Q61_cand3_aggressive.replace(cand_reverse_values, inplace=True)
df.Q61_cand4_moral.replace(cand_values, inplace=True)
df.Q61_cand6_weak.replace(cand_reverse_values, inplace=True)
df.Q61_cand7_friends.replace(cand_values, inplace=True)
df.Q61_cand13_relate.replace(cand_values, inplace=True)
df.Q61_cand14_competent.replace(cand_values, inplace=True)
df.Q61_cand15_votefor.replace(cand_values, inplace=True)
df.Q61_cand16_volunteer.replace(cand_values, inplace=True)
df.Q61_cand17_persuade.replace(cand_values, inplace=True)


# Q70
df.Q70_mess13_fair.replace(mess_values, inplace=True)
df.Q70_mess14_imprtnt.replace(mess_values, inplace=True)
df.Q70_mess15_inform.replace(mess_values, inplace=True)

# Q71
df.Q71_cand1_strong.replace(cand_values, inplace=True)
df.Q71_cand2_dishonest.replace(cand_reverse_values, inplace=True)
df.Q71_cand3_aggressive.replace(cand_reverse_values, inplace=True)
df.Q71_cand4_moral.replace(cand_values, inplace=True)
df.Q71_cand6_weak.replace(cand_reverse_values, inplace=True)
df.Q71_cand7_friends.replace(cand_values, inplace=True)
df.Q71_cand13_relate.replace(cand_values, inplace=True)
df.Q71_cand14_competent.replace(cand_values, inplace=True)
df.Q71_cand15_votefor.replace(cand_values, inplace=True)
df.Q71_cand16_volunteer.replace(cand_values, inplace=True)
df.Q71_cand17_persuade.replace(cand_values, inplace=True)


# **Combining Test Conditions**
# 

# In[5]:


print('Creating EXP_Cond column...\n')
if 'EXP_Cond' in df.columns:
    print('EXP_Cond in DataFrame')
else:
    df.insert(
        1,
        column='EXP_Cond',
        value=0)

# Coding Experimental Condition
df.loc[
    (df['Q20_mess13_fair'] >= 1) |
    (df['Q20_mess14_imprtnt'] >= 1) |
    (df['Q20_mess15_inform'] >= 1) |
    (df['Q21_cand1_strong'] >= 1) |
    (df['Q21_cand2_dishonest'] >= 1) |
    (df['Q21_cand3_aggressive'] >= 1) |
    (df['Q21_cand4_moral'] >= 1) |
    (df['Q21_cand6_weak'] >= 1) |
    (df['Q21_cand7_friends'] >= 1) |
    (df['Q21_cand13_relate'] >= 1) |
    (df['Q21_cand14_competent'] >= 1) |
    (df['Q21_cand15_votefor'] >= 1) |
    (df['Q21_cand16_volunteer'] >= 1) |
    (df['Q21_cand17_persuade'] >= 1), 
    'EXP_Cond'] = 1


df.loc[
    (df['Q30_mess13_fair'] >= 1) |
    (df['Q30_mess14_imprtnt'] >= 1) |
    (df['Q30_mess15_inform'] >= 1) |
    (df['Q31_cand1_strong'] >= 1) |
    (df['Q31_cand2_dishonest'] >= 1) |
    (df['Q31_cand3_aggressive'] >= 1) |
    (df['Q31_cand4_moral'] >= 1) |
    (df['Q31_cand6_weak'] >= 1) |
    (df['Q31_cand7_friends'] >= 1) |
    (df['Q31_cand13_relate'] >= 1) |
    (df['Q31_cand14_competent'] >= 1) |
    (df['Q31_cand15_votefor'] >= 1) |
    (df['Q31_cand16_volunteer'] >= 1) |
    (df['Q31_cand17_persuade'] >= 1), 
    'EXP_Cond'] = 2


df.loc[
    (df['Q40_mess13_fair'] >= 1) |
    (df['Q40_mess14_imprtnt'] >= 1) |
    (df['Q40_mess15_inform'] >= 1) |
    (df['Q41_cand1_strong'] >= 1) |
    (df['Q41_cand2_dishonest'] >= 1) |
    (df['Q41_cand3_aggressive'] >= 1) |
    (df['Q41_cand4_moral'] >= 1) |
    (df['Q41_cand6_weak'] >= 1) |
    (df['Q41_cand7_friends'] >= 1) |
    (df['Q41_cand13_relate'] >= 1) |
    (df['Q41_cand14_competent'] >= 1) |
    (df['Q41_cand15_votefor'] >= 1) |
    (df['Q41_cand16_volunteer'] >= 1) |
    (df['Q41_cand17_persuade'] >= 1), 
    'EXP_Cond'] = 3


df.loc[
    (df['Q50_mess13_fair'] >= 1) |
    (df['Q50_mess14_imprtnt'] >= 1) |
    (df['Q50_mess15_inform'] >= 1) |
    (df['Q51_cand1_strong'] >= 1) |
    (df['Q51_cand2_dishonest'] >= 1) |
    (df['Q51_cand3_aggressive'] >= 1) |
    (df['Q51_cand4_moral'] >= 1) |
    (df['Q51_cand6_weak'] >= 1) |
    (df['Q51_cand7_friends'] >= 1) |
    (df['Q51_cand13_relate'] >= 1) |
    (df['Q51_cand14_competent'] >= 1) |
    (df['Q51_cand15_votefor'] >= 1) |
    (df['Q51_cand16_volunteer'] >= 1) |
    (df['Q51_cand17_persuade'] >= 1), 
    'EXP_Cond'] = 4


df.loc[
    (df['Q60_mess13_fair'] >= 1) |
    (df['Q60_mess14_imprtnt'] >= 1) |
    (df['Q60_mess15_inform'] >= 1) |
    (df['Q61_cand1_strong'] >= 1) |
    (df['Q61_cand2_dishonest'] >= 1) |
    (df['Q61_cand3_aggressive'] >= 1) |
    (df['Q61_cand4_moral'] >= 1) |
    (df['Q61_cand6_weak'] >= 1) |
    (df['Q61_cand7_friends'] >= 1) |
    (df['Q61_cand13_relate'] >= 1) |
    (df['Q61_cand14_competent'] >= 1) |
    (df['Q61_cand15_votefor'] >= 1) |
    (df['Q61_cand16_volunteer'] >= 1) |
    (df['Q61_cand17_persuade'] >= 1), 
    'EXP_Cond'] = 5


df.loc[
    (df['Q70_mess13_fair'] >= 1) |
    (df['Q70_mess14_imprtnt'] >= 1) |
    (df['Q70_mess15_inform'] >= 1) |
    (df['Q71_cand1_strong'] >= 1) |
    (df['Q71_cand2_dishonest'] >= 1) |
    (df['Q71_cand3_aggressive'] >= 1) |
    (df['Q71_cand4_moral'] >= 1) |
    (df['Q71_cand6_weak'] >= 1) |
    (df['Q71_cand7_friends'] >= 1) |
    (df['Q71_cand13_relate'] >= 1) |
    (df['Q71_cand14_competent'] >= 1) |
    (df['Q71_cand15_votefor'] >= 1) |
    (df['Q71_cand16_volunteer'] >= 1) |
    (df['Q71_cand17_persuade'] >= 1), 
    'EXP_Cond'] = 6




# Human Readable Column
print('Creating EXP_Cond_HR column...\n')
if 'EXP_Cond_HR' in df.columns:
    print('EXP_Cond_HR in DataFrame')
else:
    df.insert(
        2,
        column='EXP_Cond_HR',
        value=None)


df.loc[df['EXP_Cond'] == 1, 'EXP_Cond_HR'] = 'HE-CivilPositive'
df.loc[df['EXP_Cond'] == 2, 'EXP_Cond_HR'] = 'HA-CivilPositive'
df.loc[df['EXP_Cond'] == 3, 'EXP_Cond_HR'] = 'HE-CivilNegative'
df.loc[df['EXP_Cond'] == 4, 'EXP_Cond_HR'] = 'HA-CivilNegative'
df.loc[df['EXP_Cond'] == 5, 'EXP_Cond_HR'] = 'HE-Uncivil'
df.loc[df['EXP_Cond'] == 6, 'EXP_Cond_HR'] = 'HA-Uncivil'
df.loc[df['EXP_Cond'] == 0, 'EXP_Cond_HR'] = 'NonTest'


print('Value Counts - Recoded')
print(df.EXP_Cond.value_counts())
print('\nHumanReadable')
print(df.EXP_Cond_HR.value_counts())


# ## Combining Conditions into Singular Variables
# **Final Variable Names**
# * mess13_fair
# * mess14_imprtnt
# * mess15_inform
# * cand1_strong
# * cand2_dishonest
# * cand3_aggressive
# * cand4_moral
# * cand6_weak
# * cand7_friends
# * cand13_relate
# * cand14_competent
# * cand15_votefor
# * cand16_volunteer
# * cand17_persuade

# **Message Coding**

# In[6]:


# message 13
print('Creating mess13_fair column...\n')
if 'mess13_fair' in df.columns:
    print('mess13_fair in DataFrame')
else:
    df.insert(
        5,
        column='mess13_fair',
        value=None)

df.loc[
    (df['Q20_mess13_fair'] == 1) |
    (df['Q30_mess13_fair'] == 1) |
    (df['Q40_mess13_fair'] == 1) |
    (df['Q50_mess13_fair'] == 1) |
    (df['Q60_mess13_fair'] == 1) |
    (df['Q70_mess13_fair'] == 1), 
    'mess13_fair'] = 1

df.loc[
    (df['Q20_mess13_fair'] == 2) |
    (df['Q30_mess13_fair'] == 2) |
    (df['Q40_mess13_fair'] == 2) |
    (df['Q50_mess13_fair'] == 2) |
    (df['Q60_mess13_fair'] == 2) |
    (df['Q70_mess13_fair'] == 2), 
    'mess13_fair'] = 2

df.loc[
    (df['Q20_mess13_fair'] == 3) |
    (df['Q30_mess13_fair'] == 3) |
    (df['Q40_mess13_fair'] == 3) |
    (df['Q50_mess13_fair'] == 3) |
    (df['Q60_mess13_fair'] == 3) |
    (df['Q70_mess13_fair'] == 3), 
    'mess13_fair'] = 3

df.loc[
    (df['Q20_mess13_fair'] == 4) |
    (df['Q30_mess13_fair'] == 4) |
    (df['Q40_mess13_fair'] == 4) |
    (df['Q50_mess13_fair'] == 4) |
    (df['Q60_mess13_fair'] == 4) |
    (df['Q70_mess13_fair'] == 4), 
    'mess13_fair'] = 4

print(df['mess13_fair'].value_counts(dropna=False))




# message 14
print('Creating mess14_imprtnt column...\n')
if 'mess14_imprtnt' in df.columns:
    print('mess14_imprtnt in DataFrame')
else:
    df.insert(
        5,
        column='mess14_imprtnt',
        value=None)

# mess14_imprtnt
df.loc[
    (df['Q20_mess14_imprtnt'] == 1) |
    (df['Q30_mess14_imprtnt'] == 1) |
    (df['Q40_mess14_imprtnt'] == 1) |
    (df['Q50_mess14_imprtnt'] == 1) |
    (df['Q60_mess14_imprtnt'] == 1) |
    (df['Q70_mess14_imprtnt'] == 1), 
    'mess14_imprtnt'] = 1

df.loc[
    (df['Q20_mess14_imprtnt'] == 2) |
    (df['Q30_mess14_imprtnt'] == 2) |
    (df['Q40_mess14_imprtnt'] == 2) |
    (df['Q50_mess14_imprtnt'] == 2) |
    (df['Q60_mess14_imprtnt'] == 2) |
    (df['Q70_mess14_imprtnt'] == 2), 
    'mess14_imprtnt'] = 2

df.loc[
    (df['Q20_mess14_imprtnt'] == 3) |
    (df['Q30_mess14_imprtnt'] == 3) |
    (df['Q40_mess14_imprtnt'] == 3) |
    (df['Q50_mess14_imprtnt'] == 3) |
    (df['Q60_mess14_imprtnt'] == 3) |
    (df['Q70_mess14_imprtnt'] == 3), 
    'mess14_imprtnt'] = 3

df.loc[
    (df['Q20_mess14_imprtnt'] == 4) |
    (df['Q30_mess14_imprtnt'] == 4) |
    (df['Q40_mess14_imprtnt'] == 4) |
    (df['Q50_mess14_imprtnt'] == 4) |
    (df['Q60_mess14_imprtnt'] == 4) |
    (df['Q70_mess14_imprtnt'] == 4), 
    'mess14_imprtnt'] = 4


print(df['mess14_imprtnt'].value_counts(dropna=False))



# message 15
print('Creating mess15_inform column...\n')
if 'mess15_inform' in df.columns:
    print('mess15_inform in DataFrame')
else:
    df.insert(
        5,
        column='mess15_inform',
        value=None)

# mess15_inform
df.loc[
    (df['Q20_mess15_inform'] == 1) |
    (df['Q30_mess15_inform'] == 1) |
    (df['Q40_mess15_inform'] == 1) |
    (df['Q50_mess15_inform'] == 1) |
    (df['Q60_mess15_inform'] == 1) |
    (df['Q70_mess15_inform'] == 1), 
    'mess15_inform'] = 1

df.loc[
    (df['Q20_mess15_inform'] == 2) |
    (df['Q30_mess15_inform'] == 2) |
    (df['Q40_mess15_inform'] == 2) |
    (df['Q50_mess15_inform'] == 2) |
    (df['Q60_mess15_inform'] == 2) |
    (df['Q70_mess15_inform'] == 2), 
    'mess15_inform'] = 2

df.loc[
    (df['Q20_mess15_inform'] == 3) |
    (df['Q30_mess15_inform'] == 3) |
    (df['Q40_mess15_inform'] == 3) |
    (df['Q50_mess15_inform'] == 3) |
    (df['Q60_mess15_inform'] == 3) |
    (df['Q70_mess15_inform'] == 3), 
    'mess15_inform'] = 3

df.loc[
    (df['Q20_mess15_inform'] == 4) |
    (df['Q30_mess15_inform'] == 4) |
    (df['Q40_mess15_inform'] == 4) |
    (df['Q50_mess15_inform'] == 4) |
    (df['Q60_mess15_inform'] == 4) |
    (df['Q70_mess15_inform'] == 4), 
    'mess15_inform'] = 4

print(df['mess15_inform'].value_counts(dropna=False))


# **Candidate Coding**

# In[7]:


print('Creating cand1_strong column...\n')
if 'cand1_strong' in df.columns:
    print('cand1_strong in DataFrame')
else:
    df.insert(
        5,
        column='cand1_strong',
        value=None)

# candidate strength
df.loc[
    (df['Q21_cand1_strong'] == 1) |
    (df['Q31_cand1_strong'] == 1) |
    (df['Q41_cand1_strong'] == 1) |
    (df['Q51_cand1_strong'] == 1) |
    (df['Q61_cand1_strong'] == 1) |
    (df['Q71_cand1_strong'] == 1), 
    'cand1_strong'] = 1

df.loc[
    (df['Q21_cand1_strong'] == 2) |
    (df['Q31_cand1_strong'] == 2) |
    (df['Q41_cand1_strong'] == 2) |
    (df['Q51_cand1_strong'] == 2) |
    (df['Q61_cand1_strong'] == 2) |
    (df['Q71_cand1_strong'] == 2), 
    'cand1_strong'] = 2

df.loc[
    (df['Q21_cand1_strong'] == 3) |
    (df['Q31_cand1_strong'] == 3) |
    (df['Q41_cand1_strong'] == 3) |
    (df['Q51_cand1_strong'] == 3) |
    (df['Q61_cand1_strong'] == 3) |
    (df['Q71_cand1_strong'] == 3), 
    'cand1_strong'] = 3

df.loc[
    (df['Q21_cand1_strong'] == 4) |
    (df['Q31_cand1_strong'] == 4) |
    (df['Q41_cand1_strong'] == 4) |
    (df['Q51_cand1_strong'] == 4) |
    (df['Q61_cand1_strong'] == 4) |
    (df['Q71_cand1_strong'] == 4), 
    'cand1_strong'] = 4

df.loc[
    (df['Q21_cand1_strong'] == 5) |
    (df['Q31_cand1_strong'] == 5) |
    (df['Q41_cand1_strong'] == 5) |
    (df['Q51_cand1_strong'] == 5) |
    (df['Q61_cand1_strong'] == 5) |
    (df['Q71_cand1_strong'] == 5), 
    'cand1_strong'] = 5

df.loc[
    (df['Q21_cand1_strong'] == 6) |
    (df['Q31_cand1_strong'] == 6) |
    (df['Q41_cand1_strong'] == 6) |
    (df['Q51_cand1_strong'] == 6) |
    (df['Q61_cand1_strong'] == 6) |
    (df['Q71_cand1_strong'] == 6), 
    'cand1_strong'] = 6

df.loc[
    (df['Q21_cand1_strong'] == 7) |
    (df['Q31_cand1_strong'] == 7) |
    (df['Q41_cand1_strong'] == 7) |
    (df['Q51_cand1_strong'] == 7) |
    (df['Q61_cand1_strong'] == 7) |
    (df['Q71_cand1_strong'] == 7), 
    'cand1_strong'] = 7



# candidate dishonesty
print('Creating cand2_dishonest column...\n')
if 'cand2_dishonest' in df.columns:
    print('cand2_dishonest in DataFrame')
else:
    df.insert(
        5,
        column='cand2_dishonest',
        value=None)

# cand2_dishonest
df.loc[
    (df['Q21_cand2_dishonest'] == 1) |
    (df['Q31_cand2_dishonest'] == 1) |
    (df['Q41_cand2_dishonest'] == 1) |
    (df['Q51_cand2_dishonest'] == 1) |
    (df['Q61_cand2_dishonest'] == 1) |
    (df['Q71_cand2_dishonest'] == 1), 
    'cand2_dishonest'] = 1

df.loc[
    (df['Q21_cand2_dishonest'] == 2) |
    (df['Q31_cand2_dishonest'] == 2) |
    (df['Q41_cand2_dishonest'] == 2) |
    (df['Q51_cand2_dishonest'] == 2) |
    (df['Q61_cand2_dishonest'] == 2) |
    (df['Q71_cand2_dishonest'] == 2), 
    'cand2_dishonest'] = 2

df.loc[
    (df['Q21_cand2_dishonest'] == 3) |
    (df['Q31_cand2_dishonest'] == 3) |
    (df['Q41_cand2_dishonest'] == 3) |
    (df['Q51_cand2_dishonest'] == 3) |
    (df['Q61_cand2_dishonest'] == 3) |
    (df['Q71_cand2_dishonest'] == 3), 
    'cand2_dishonest'] = 3

df.loc[
    (df['Q21_cand2_dishonest'] == 4) |
    (df['Q31_cand2_dishonest'] == 4) |
    (df['Q41_cand2_dishonest'] == 4) |
    (df['Q51_cand2_dishonest'] == 4) |
    (df['Q61_cand2_dishonest'] == 4) |
    (df['Q71_cand2_dishonest'] == 4), 
    'cand2_dishonest'] = 4

df.loc[
    (df['Q21_cand2_dishonest'] == 5) |
    (df['Q31_cand2_dishonest'] == 5) |
    (df['Q41_cand2_dishonest'] == 5) |
    (df['Q51_cand2_dishonest'] == 5) |
    (df['Q61_cand2_dishonest'] == 5) |
    (df['Q71_cand2_dishonest'] == 5), 
    'cand2_dishonest'] = 5

df.loc[
    (df['Q21_cand2_dishonest'] == 6) |
    (df['Q31_cand2_dishonest'] == 6) |
    (df['Q41_cand2_dishonest'] == 6) |
    (df['Q51_cand2_dishonest'] == 6) |
    (df['Q61_cand2_dishonest'] == 6) |
    (df['Q71_cand2_dishonest'] == 6), 
    'cand2_dishonest'] = 6

df.loc[
    (df['Q21_cand2_dishonest'] == 7) |
    (df['Q31_cand2_dishonest'] == 7) |
    (df['Q41_cand2_dishonest'] == 7) |
    (df['Q51_cand2_dishonest'] == 7) |
    (df['Q61_cand2_dishonest'] == 7) |
    (df['Q71_cand2_dishonest'] == 7), 
    'cand2_dishonest'] = 7


# candidate aggression
print('Creating cand3_aggressive column...\n')
if 'cand3_aggressive' in df.columns:
    print('cand3_aggressive in DataFrame')
else:
    df.insert(
        5,
        column='cand3_aggressive',
        value=None)

df.loc[
    (df['Q21_cand3_aggressive'] == 1) |
    (df['Q31_cand3_aggressive'] == 1) |
    (df['Q41_cand3_aggressive'] == 1) |
    (df['Q51_cand3_aggressive'] == 1) |
    (df['Q61_cand3_aggressive'] == 1) |
    (df['Q71_cand3_aggressive'] == 1), 
    'cand3_aggressive'] = 1

df.loc[
    (df['Q21_cand3_aggressive'] == 2) |
    (df['Q31_cand3_aggressive'] == 2) |
    (df['Q41_cand3_aggressive'] == 2) |
    (df['Q51_cand3_aggressive'] == 2) |
    (df['Q61_cand3_aggressive'] == 2) |
    (df['Q71_cand3_aggressive'] == 2), 
    'cand3_aggressive'] = 2

df.loc[
    (df['Q21_cand3_aggressive'] == 3) |
    (df['Q31_cand3_aggressive'] == 3) |
    (df['Q41_cand3_aggressive'] == 3) |
    (df['Q51_cand3_aggressive'] == 3) |
    (df['Q61_cand3_aggressive'] == 3) |
    (df['Q71_cand3_aggressive'] == 3), 
    'cand3_aggressive'] = 3

df.loc[
    (df['Q21_cand3_aggressive'] == 4) |
    (df['Q31_cand3_aggressive'] == 4) |
    (df['Q41_cand3_aggressive'] == 4) |
    (df['Q51_cand3_aggressive'] == 4) |
    (df['Q61_cand3_aggressive'] == 4) |
    (df['Q71_cand3_aggressive'] == 4), 
    'cand3_aggressive'] = 4

df.loc[
    (df['Q21_cand3_aggressive'] == 5) |
    (df['Q31_cand3_aggressive'] == 5) |
    (df['Q41_cand3_aggressive'] == 5) |
    (df['Q51_cand3_aggressive'] == 5) |
    (df['Q61_cand3_aggressive'] == 5) |
    (df['Q71_cand3_aggressive'] == 5), 
    'cand3_aggressive'] = 5

df.loc[
    (df['Q21_cand3_aggressive'] == 6) |
    (df['Q31_cand3_aggressive'] == 6) |
    (df['Q41_cand3_aggressive'] == 6) |
    (df['Q51_cand3_aggressive'] == 6) |
    (df['Q61_cand3_aggressive'] == 6) |
    (df['Q71_cand3_aggressive'] == 6), 
    'cand3_aggressive'] = 6

df.loc[
    (df['Q21_cand3_aggressive'] == 7) |
    (df['Q31_cand3_aggressive'] == 7) |
    (df['Q41_cand3_aggressive'] == 7) |
    (df['Q51_cand3_aggressive'] == 7) |
    (df['Q61_cand3_aggressive'] == 7) |
    (df['Q71_cand3_aggressive'] == 7), 
    'cand3_aggressive'] = 7

# candidate morality
print('Creating cand4_moral column...\n')
if 'cand4_moral' in df.columns:
    print('cand4_moral in DataFrame')
else:
    df.insert(
        5,
        column='cand4_moral',
        value=None)

df.loc[
    (df['Q21_cand4_moral'] == 1) |
    (df['Q31_cand4_moral'] == 1) |
    (df['Q41_cand4_moral'] == 1) |
    (df['Q51_cand4_moral'] == 1) |
    (df['Q61_cand4_moral'] == 1) |
    (df['Q71_cand4_moral'] == 1), 
    'cand4_moral'] = 1

df.loc[
    (df['Q21_cand4_moral'] == 2) |
    (df['Q31_cand4_moral'] == 2) |
    (df['Q41_cand4_moral'] == 2) |
    (df['Q51_cand4_moral'] == 2) |
    (df['Q61_cand4_moral'] == 2) |
    (df['Q71_cand4_moral'] == 2), 
    'cand4_moral'] = 2

df.loc[
    (df['Q21_cand4_moral'] == 3) |
    (df['Q31_cand4_moral'] == 3) |
    (df['Q41_cand4_moral'] == 3) |
    (df['Q51_cand4_moral'] == 3) |
    (df['Q61_cand4_moral'] == 3) |
    (df['Q71_cand4_moral'] == 3), 
    'cand4_moral'] = 3

df.loc[
    (df['Q21_cand4_moral'] == 4) |
    (df['Q31_cand4_moral'] == 4) |
    (df['Q41_cand4_moral'] == 4) |
    (df['Q51_cand4_moral'] == 4) |
    (df['Q61_cand4_moral'] == 4) |
    (df['Q71_cand4_moral'] == 4), 
    'cand4_moral'] = 4

df.loc[
    (df['Q21_cand4_moral'] == 5) |
    (df['Q31_cand4_moral'] == 5) |
    (df['Q41_cand4_moral'] == 5) |
    (df['Q51_cand4_moral'] == 5) |
    (df['Q61_cand4_moral'] == 5) |
    (df['Q71_cand4_moral'] == 5), 
    'cand4_moral'] = 5

df.loc[
    (df['Q21_cand4_moral'] == 6) |
    (df['Q31_cand4_moral'] == 6) |
    (df['Q41_cand4_moral'] == 6) |
    (df['Q51_cand4_moral'] == 6) |
    (df['Q61_cand4_moral'] == 6) |
    (df['Q71_cand4_moral'] == 6), 
    'cand4_moral'] = 6

df.loc[
    (df['Q21_cand4_moral'] == 7) |
    (df['Q31_cand4_moral'] == 7) |
    (df['Q41_cand4_moral'] == 7) |
    (df['Q51_cand4_moral'] == 7) |
    (df['Q61_cand4_moral'] == 7) |
    (df['Q71_cand4_moral'] == 7), 
    'cand4_moral'] = 7


# candididate weakness
print('Creating cand6_weak column...\n')
if 'cand6_weak' in df.columns:
    print('cand6_weak in DataFrame')
else:
    df.insert(
        5,
        column='cand6_weak',
        value=None)

df.loc[
    (df['Q21_cand6_weak'] == 1) |
    (df['Q31_cand6_weak'] == 1) |
    (df['Q41_cand6_weak'] == 1) |
    (df['Q51_cand6_weak'] == 1) |
    (df['Q61_cand6_weak'] == 1) |
    (df['Q71_cand6_weak'] == 1), 
    'cand6_weak'] = 1

df.loc[
    (df['Q21_cand6_weak'] == 2) |
    (df['Q31_cand6_weak'] == 2) |
    (df['Q41_cand6_weak'] == 2) |
    (df['Q51_cand6_weak'] == 2) |
    (df['Q61_cand6_weak'] == 2) |
    (df['Q71_cand6_weak'] == 2), 
    'cand6_weak'] = 2

df.loc[
    (df['Q21_cand6_weak'] == 3) |
    (df['Q31_cand6_weak'] == 3) |
    (df['Q41_cand6_weak'] == 3) |
    (df['Q51_cand6_weak'] == 3) |
    (df['Q61_cand6_weak'] == 3) |
    (df['Q71_cand6_weak'] == 3), 
    'cand6_weak'] = 3

df.loc[
    (df['Q21_cand6_weak'] == 4) |
    (df['Q31_cand6_weak'] == 4) |
    (df['Q41_cand6_weak'] == 4) |
    (df['Q51_cand6_weak'] == 4) |
    (df['Q61_cand6_weak'] == 4) |
    (df['Q71_cand6_weak'] == 4), 
    'cand6_weak'] = 4

df.loc[
    (df['Q21_cand6_weak'] == 5) |
    (df['Q31_cand6_weak'] == 5) |
    (df['Q41_cand6_weak'] == 5) |
    (df['Q51_cand6_weak'] == 5) |
    (df['Q61_cand6_weak'] == 5) |
    (df['Q71_cand6_weak'] == 5), 
    'cand6_weak'] = 5

df.loc[
    (df['Q21_cand6_weak'] == 6) |
    (df['Q31_cand6_weak'] == 6) |
    (df['Q41_cand6_weak'] == 6) |
    (df['Q51_cand6_weak'] == 6) |
    (df['Q61_cand6_weak'] == 6) |
    (df['Q71_cand6_weak'] == 6), 
    'cand6_weak'] = 6

df.loc[
    (df['Q21_cand6_weak'] == 7) |
    (df['Q31_cand6_weak'] == 7) |
    (df['Q41_cand6_weak'] == 7) |
    (df['Q51_cand6_weak'] == 7) |
    (df['Q61_cand6_weak'] == 7) |
    (df['Q71_cand6_weak'] == 7), 
    'cand6_weak'] = 7


# candidate friendship
print('Creating cand7_friends column...\n')
if 'cand7_friends' in df.columns:
    print('cand7_friends in DataFrame')
else:
    df.insert(
        5,
        column='cand7_friends',
        value=None)

df.loc[
    (df['Q21_cand7_friends'] == 1) |
    (df['Q31_cand7_friends'] == 1) |
    (df['Q41_cand7_friends'] == 1) |
    (df['Q51_cand7_friends'] == 1) |
    (df['Q61_cand7_friends'] == 1) |
    (df['Q71_cand7_friends'] == 1), 
    'cand7_friends'] = 1

df.loc[
    (df['Q21_cand7_friends'] == 2) |
    (df['Q31_cand7_friends'] == 2) |
    (df['Q41_cand7_friends'] == 2) |
    (df['Q51_cand7_friends'] == 2) |
    (df['Q61_cand7_friends'] == 2) |
    (df['Q71_cand7_friends'] == 2), 
    'cand7_friends'] = 2

df.loc[
    (df['Q21_cand7_friends'] == 3) |
    (df['Q31_cand7_friends'] == 3) |
    (df['Q41_cand7_friends'] == 3) |
    (df['Q51_cand7_friends'] == 3) |
    (df['Q61_cand7_friends'] == 3) |
    (df['Q71_cand7_friends'] == 3), 
    'cand7_friends'] = 3

df.loc[
    (df['Q21_cand7_friends'] == 4) |
    (df['Q31_cand7_friends'] == 4) |
    (df['Q41_cand7_friends'] == 4) |
    (df['Q51_cand7_friends'] == 4) |
    (df['Q61_cand7_friends'] == 4) |
    (df['Q71_cand7_friends'] == 4), 
    'cand7_friends'] = 4

df.loc[
    (df['Q21_cand7_friends'] == 5) |
    (df['Q31_cand7_friends'] == 5) |
    (df['Q41_cand7_friends'] == 5) |
    (df['Q51_cand7_friends'] == 5) |
    (df['Q61_cand7_friends'] == 5) |
    (df['Q71_cand7_friends'] == 5), 
    'cand7_friends'] = 5

df.loc[
    (df['Q21_cand7_friends'] == 6) |
    (df['Q31_cand7_friends'] == 6) |
    (df['Q41_cand7_friends'] == 6) |
    (df['Q51_cand7_friends'] == 6) |
    (df['Q61_cand7_friends'] == 6) |
    (df['Q71_cand7_friends'] == 6), 
    'cand7_friends'] = 6

df.loc[
    (df['Q21_cand7_friends'] == 7) |
    (df['Q31_cand7_friends'] == 7) |
    (df['Q41_cand7_friends'] == 7) |
    (df['Q51_cand7_friends'] == 7) |
    (df['Q61_cand7_friends'] == 7) |
    (df['Q71_cand7_friends'] == 7), 
    'cand7_friends'] = 7


# candidate relatability
print('Creating cand13_relate column...\n')
if 'cand13_relate' in df.columns:
    print('cand13_relate in DataFrame')
else:
    df.insert(
        5,
        column='cand13_relate',
        value=None)

df.loc[
    (df['Q21_cand13_relate'] == 1) |
    (df['Q31_cand13_relate'] == 1) |
    (df['Q41_cand13_relate'] == 1) |
    (df['Q51_cand13_relate'] == 1) |
    (df['Q61_cand13_relate'] == 1) |
    (df['Q71_cand13_relate'] == 1), 
    'cand13_relate'] = 1

df.loc[
    (df['Q21_cand13_relate'] == 2) |
    (df['Q31_cand13_relate'] == 2) |
    (df['Q41_cand13_relate'] == 2) |
    (df['Q51_cand13_relate'] == 2) |
    (df['Q61_cand13_relate'] == 2) |
    (df['Q71_cand13_relate'] == 2), 
    'cand13_relate'] = 2

df.loc[
    (df['Q21_cand13_relate'] == 3) |
    (df['Q31_cand13_relate'] == 3) |
    (df['Q41_cand13_relate'] == 3) |
    (df['Q51_cand13_relate'] == 3) |
    (df['Q61_cand13_relate'] == 3) |
    (df['Q71_cand13_relate'] == 3), 
    'cand13_relate'] = 3

df.loc[
    (df['Q21_cand13_relate'] == 4) |
    (df['Q31_cand13_relate'] == 4) |
    (df['Q41_cand13_relate'] == 4) |
    (df['Q51_cand13_relate'] == 4) |
    (df['Q61_cand13_relate'] == 4) |
    (df['Q71_cand13_relate'] == 4), 
    'cand13_relate'] = 4

df.loc[
    (df['Q21_cand13_relate'] == 5) |
    (df['Q31_cand13_relate'] == 5) |
    (df['Q41_cand13_relate'] == 5) |
    (df['Q51_cand13_relate'] == 5) |
    (df['Q61_cand13_relate'] == 5) |
    (df['Q71_cand13_relate'] == 5), 
    'cand13_relate'] = 5

df.loc[
    (df['Q21_cand13_relate'] == 6) |
    (df['Q31_cand13_relate'] == 6) |
    (df['Q41_cand13_relate'] == 6) |
    (df['Q51_cand13_relate'] == 6) |
    (df['Q61_cand13_relate'] == 6) |
    (df['Q71_cand13_relate'] == 6), 
    'cand13_relate'] = 6

df.loc[
    (df['Q21_cand13_relate'] == 7) |
    (df['Q31_cand13_relate'] == 7) |
    (df['Q41_cand13_relate'] == 7) |
    (df['Q51_cand13_relate'] == 7) |
    (df['Q61_cand13_relate'] == 7) |
    (df['Q71_cand13_relate'] == 7), 
    'cand13_relate'] = 7


# candidate competence
print('Creating cand14_competent column...\n')
if 'cand14_competent' in df.columns:
    print('cand14_competent in DataFrame')
else:
    df.insert(
        5,
        column='cand14_competent',
        value=None)

df.loc[
    (df['Q21_cand14_competent'] == 1) |
    (df['Q31_cand14_competent'] == 1) |
    (df['Q41_cand14_competent'] == 1) |
    (df['Q51_cand14_competent'] == 1) |
    (df['Q61_cand14_competent'] == 1) |
    (df['Q71_cand14_competent'] == 1), 
    'cand14_competent'] = 1

df.loc[
    (df['Q21_cand14_competent'] == 2) |
    (df['Q31_cand14_competent'] == 2) |
    (df['Q41_cand14_competent'] == 2) |
    (df['Q51_cand14_competent'] == 2) |
    (df['Q61_cand14_competent'] == 2) |
    (df['Q71_cand14_competent'] == 2), 
    'cand14_competent'] = 2

df.loc[
    (df['Q21_cand14_competent'] == 3) |
    (df['Q31_cand14_competent'] == 3) |
    (df['Q41_cand14_competent'] == 3) |
    (df['Q51_cand14_competent'] == 3) |
    (df['Q61_cand14_competent'] == 3) |
    (df['Q71_cand14_competent'] == 3), 
    'cand14_competent'] = 3

df.loc[
    (df['Q21_cand14_competent'] == 4) |
    (df['Q31_cand14_competent'] == 4) |
    (df['Q41_cand14_competent'] == 4) |
    (df['Q51_cand14_competent'] == 4) |
    (df['Q61_cand14_competent'] == 4) |
    (df['Q71_cand14_competent'] == 4), 
    'cand14_competent'] = 4

df.loc[
    (df['Q21_cand14_competent'] == 5) |
    (df['Q31_cand14_competent'] == 5) |
    (df['Q41_cand14_competent'] == 5) |
    (df['Q51_cand14_competent'] == 5) |
    (df['Q61_cand14_competent'] == 5) |
    (df['Q71_cand14_competent'] == 5), 
    'cand14_competent'] = 5

df.loc[
    (df['Q21_cand14_competent'] == 6) |
    (df['Q31_cand14_competent'] == 6) |
    (df['Q41_cand14_competent'] == 6) |
    (df['Q51_cand14_competent'] == 6) |
    (df['Q61_cand14_competent'] == 6) |
    (df['Q71_cand14_competent'] == 6), 
    'cand14_competent'] = 6

df.loc[
    (df['Q21_cand14_competent'] == 7) |
    (df['Q31_cand14_competent'] == 7) |
    (df['Q41_cand14_competent'] == 7) |
    (df['Q51_cand14_competent'] == 7) |
    (df['Q61_cand14_competent'] == 7) |
    (df['Q71_cand14_competent'] == 7), 
    'cand14_competent'] = 7


# candidate - vote for
print('Creating cand15_votefor column...\n')
if 'cand15_votefor' in df.columns:
    print('cand15_votefor in DataFrame')
else:
    df.insert(
        5,
        column='cand15_votefor',
        value=None)

df.loc[
    (df['Q21_cand15_votefor'] == 1) |
    (df['Q31_cand15_votefor'] == 1) |
    (df['Q41_cand15_votefor'] == 1) |
    (df['Q51_cand15_votefor'] == 1) |
    (df['Q61_cand15_votefor'] == 1) |
    (df['Q71_cand15_votefor'] == 1), 
    'cand15_votefor'] = 1

df.loc[
    (df['Q21_cand15_votefor'] == 2) |
    (df['Q31_cand15_votefor'] == 2) |
    (df['Q41_cand15_votefor'] == 2) |
    (df['Q51_cand15_votefor'] == 2) |
    (df['Q61_cand15_votefor'] == 2) |
    (df['Q71_cand15_votefor'] == 2), 
    'cand15_votefor'] = 2

df.loc[
    (df['Q21_cand15_votefor'] == 3) |
    (df['Q31_cand15_votefor'] == 3) |
    (df['Q41_cand15_votefor'] == 3) |
    (df['Q51_cand15_votefor'] == 3) |
    (df['Q61_cand15_votefor'] == 3) |
    (df['Q71_cand15_votefor'] == 3), 
    'cand15_votefor'] = 3

df.loc[
    (df['Q21_cand15_votefor'] == 4) |
    (df['Q31_cand15_votefor'] == 4) |
    (df['Q41_cand15_votefor'] == 4) |
    (df['Q51_cand15_votefor'] == 4) |
    (df['Q61_cand15_votefor'] == 4) |
    (df['Q71_cand15_votefor'] == 4), 
    'cand15_votefor'] = 4

df.loc[
    (df['Q21_cand15_votefor'] == 5) |
    (df['Q31_cand15_votefor'] == 5) |
    (df['Q41_cand15_votefor'] == 5) |
    (df['Q51_cand15_votefor'] == 5) |
    (df['Q61_cand15_votefor'] == 5) |
    (df['Q71_cand15_votefor'] == 5), 
    'cand15_votefor'] = 5

df.loc[
    (df['Q21_cand15_votefor'] == 6) |
    (df['Q31_cand15_votefor'] == 6) |
    (df['Q41_cand15_votefor'] == 6) |
    (df['Q51_cand15_votefor'] == 6) |
    (df['Q61_cand15_votefor'] == 6) |
    (df['Q71_cand15_votefor'] == 6), 
    'cand15_votefor'] = 6

df.loc[
    (df['Q21_cand15_votefor'] == 7) |
    (df['Q31_cand15_votefor'] == 7) |
    (df['Q41_cand15_votefor'] == 7) |
    (df['Q51_cand15_votefor'] == 7) |
    (df['Q61_cand15_votefor'] == 7) |
    (df['Q71_cand15_votefor'] == 7), 
    'cand15_votefor'] = 7


# candidate - volunteer
print('Creating cand16_volunteer column...\n')
if 'cand16_volunteer' in df.columns:
    print('cand16_volunteer in DataFrame')
else:
    df.insert(
        5,
        column='cand16_volunteer',
        value=None)

df.loc[
    (df['Q21_cand16_volunteer'] == 1) |
    (df['Q31_cand16_volunteer'] == 1) |
    (df['Q41_cand16_volunteer'] == 1) |
    (df['Q51_cand16_volunteer'] == 1) |
    (df['Q61_cand16_volunteer'] == 1) |
    (df['Q71_cand16_volunteer'] == 1), 
    'cand16_volunteer'] = 1

df.loc[
    (df['Q21_cand16_volunteer'] == 2) |
    (df['Q31_cand16_volunteer'] == 2) |
    (df['Q41_cand16_volunteer'] == 2) |
    (df['Q51_cand16_volunteer'] == 2) |
    (df['Q61_cand16_volunteer'] == 2) |
    (df['Q71_cand16_volunteer'] == 2), 
    'cand16_volunteer'] = 2

df.loc[
    (df['Q21_cand16_volunteer'] == 3) |
    (df['Q31_cand16_volunteer'] == 3) |
    (df['Q41_cand16_volunteer'] == 3) |
    (df['Q51_cand16_volunteer'] == 3) |
    (df['Q61_cand16_volunteer'] == 3) |
    (df['Q71_cand16_volunteer'] == 3), 
    'cand16_volunteer'] = 3

df.loc[
    (df['Q21_cand16_volunteer'] == 4) |
    (df['Q31_cand16_volunteer'] == 4) |
    (df['Q41_cand16_volunteer'] == 4) |
    (df['Q51_cand16_volunteer'] == 4) |
    (df['Q61_cand16_volunteer'] == 4) |
    (df['Q71_cand16_volunteer'] == 4), 
    'cand16_volunteer'] = 4

df.loc[
    (df['Q21_cand16_volunteer'] == 5) |
    (df['Q31_cand16_volunteer'] == 5) |
    (df['Q41_cand16_volunteer'] == 5) |
    (df['Q51_cand16_volunteer'] == 5) |
    (df['Q61_cand16_volunteer'] == 5) |
    (df['Q71_cand16_volunteer'] == 5), 
    'cand16_volunteer'] = 5

df.loc[
    (df['Q21_cand16_volunteer'] == 6) |
    (df['Q31_cand16_volunteer'] == 6) |
    (df['Q41_cand16_volunteer'] == 6) |
    (df['Q51_cand16_volunteer'] == 6) |
    (df['Q61_cand16_volunteer'] == 6) |
    (df['Q71_cand16_volunteer'] == 6), 
    'cand16_volunteer'] = 6

df.loc[
    (df['Q21_cand16_volunteer'] == 7) |
    (df['Q31_cand16_volunteer'] == 7) |
    (df['Q41_cand16_volunteer'] == 7) |
    (df['Q51_cand16_volunteer'] == 7) |
    (df['Q61_cand16_volunteer'] == 7) |
    (df['Q71_cand16_volunteer'] == 7), 
    'cand16_volunteer'] = 7


# candidate persuasion
print('Creating cand17_persuade column...\n')
if 'cand17_persuade' in df.columns:
    print('cand17_persuade in DataFrame')
else:
    df.insert(
        5,
        column='cand17_persuade',
        value=None)

df.loc[
    (df['Q21_cand17_persuade'] == 1) |
    (df['Q31_cand17_persuade'] == 1) |
    (df['Q41_cand17_persuade'] == 1) |
    (df['Q51_cand17_persuade'] == 1) |
    (df['Q61_cand17_persuade'] == 1) |
    (df['Q71_cand17_persuade'] == 1), 
    'cand17_persuade'] = 1

df.loc[
    (df['Q21_cand17_persuade'] == 2) |
    (df['Q31_cand17_persuade'] == 2) |
    (df['Q41_cand17_persuade'] == 2) |
    (df['Q51_cand17_persuade'] == 2) |
    (df['Q61_cand17_persuade'] == 2) |
    (df['Q71_cand17_persuade'] == 2), 
    'cand17_persuade'] = 2

df.loc[
    (df['Q21_cand17_persuade'] == 3) |
    (df['Q31_cand17_persuade'] == 3) |
    (df['Q41_cand17_persuade'] == 3) |
    (df['Q51_cand17_persuade'] == 3) |
    (df['Q61_cand17_persuade'] == 3) |
    (df['Q71_cand17_persuade'] == 3), 
    'cand17_persuade'] = 3

df.loc[
    (df['Q21_cand17_persuade'] == 4) |
    (df['Q31_cand17_persuade'] == 4) |
    (df['Q41_cand17_persuade'] == 4) |
    (df['Q51_cand17_persuade'] == 4) |
    (df['Q61_cand17_persuade'] == 4) |
    (df['Q71_cand17_persuade'] == 4), 
    'cand17_persuade'] = 4

df.loc[
    (df['Q21_cand17_persuade'] == 5) |
    (df['Q31_cand17_persuade'] == 5) |
    (df['Q41_cand17_persuade'] == 5) |
    (df['Q51_cand17_persuade'] == 5) |
    (df['Q61_cand17_persuade'] == 5) |
    (df['Q71_cand17_persuade'] == 5), 
    'cand17_persuade'] = 5

df.loc[
    (df['Q21_cand17_persuade'] == 6) |
    (df['Q31_cand17_persuade'] == 6) |
    (df['Q41_cand17_persuade'] == 6) |
    (df['Q51_cand17_persuade'] == 6) |
    (df['Q61_cand17_persuade'] == 6) |
    (df['Q71_cand17_persuade'] == 6), 
    'cand17_persuade'] = 6

df.loc[
    (df['Q21_cand17_persuade'] == 7) |
    (df['Q31_cand17_persuade'] == 7) |
    (df['Q41_cand17_persuade'] == 7) |
    (df['Q51_cand17_persuade'] == 7) |
    (df['Q61_cand17_persuade'] == 7) |
    (df['Q71_cand17_persuade'] == 7), 
    'cand17_persuade'] = 7


# In[8]:


df_all = df.loc[
    :, ['ResponseId',
        'Age',
        'Ethnicity',
        'Sex',
        'EXP_Cond',
        'EXP_Cond_HR',
        'sdo1_Pro_Trait_Dom1',
        'sdo13_Pro_Trait_Dom2',
        'sdo6_Con_Trait_Dom2',
        'sdo2_Con_Trait_Dom1',
        'sdo7_Pro_Trait_AntiEgal1',
        'sdo3_Pro_Trait_AntiEgal2',
        'sdo4_Con_Trait_AntiEgal2',
        'sdo14_Con_Trait_AntiEgal1',
        'ideol2_social',
        'ideol4_econ',
        'ideol3_self',
        'trust13_officials',
        'trust6_nocare',
        'trust2_nosay',
        'pol_interest',
        'pol_vote',
        'cand17_persuade',
        'cand16_volunteer',
        'cand15_votefor',
        'cand14_competent',
        'cand13_relate',
        'cand7_friends',
        'cand6_weak',
        'cand4_moral',
        'cand3_aggressive',
        'cand2_dishonest',
        'cand1_strong',
        'mess15_inform',
        'mess14_imprtnt',
        'mess13_fair',
        'Q20_mess14_imprtnt',
        'Q20_mess15_inform',
        'Q20_mess13_fair',
        'Q21_cand1_strong',
        'Q21_cand13_relate',
        'Q21_cand6_weak',
        'Q21_cand2_dishonest',
        'Q21_cand7_friends',
        'Q21_cand3_aggressive',
        'Q21_cand4_moral',
        'Q21_cand14_competent',
        'Q21_cand15_votefor',
        'Q21_cand16_volunteer',
        'Q21_cand17_persuade',
        'Q30_mess14_imprtnt',
        'Q30_mess15_inform',
        'Q30_mess13_fair',
        'Q31_cand1_strong',
        'Q31_cand13_relate',
        'Q31_cand6_weak',
        'Q31_cand2_dishonest',
        'Q31_cand7_friends',
        'Q31_cand3_aggressive',
        'Q31_cand4_moral',
        'Q31_cand14_competent',
        'Q31_cand15_votefor',
        'Q31_cand16_volunteer',
        'Q31_cand17_persuade',
        'Q40_mess14_imprtnt',
        'Q40_mess15_inform',
        'Q40_mess13_fair',
        'Q41_cand1_strong',
        'Q41_cand13_relate',
        'Q41_cand6_weak',
        'Q41_cand2_dishonest',
        'Q41_cand7_friends',
        'Q41_cand3_aggressive',
        'Q41_cand4_moral',
        'Q41_cand14_competent',
        'Q41_cand15_votefor',
        'Q41_cand16_volunteer',
        'Q41_cand17_persuade',
        'Q50_mess14_imprtnt',
        'Q50_mess15_inform',
        'Q50_mess13_fair',
        'Q51_cand1_strong',
        'Q51_cand13_relate',
        'Q51_cand6_weak',
        'Q51_cand2_dishonest',
        'Q51_cand7_friends',
        'Q51_cand3_aggressive',
        'Q51_cand4_moral',
        'Q51_cand14_competent',
        'Q51_cand15_votefor',
        'Q51_cand16_volunteer',
        'Q51_cand17_persuade',
        'Q60_mess14_imprtnt',
        'Q60_mess15_inform',
        'Q60_mess13_fair',
        'Q61_cand1_strong',
        'Q61_cand13_relate',
        'Q61_cand6_weak',
        'Q61_cand2_dishonest',
        'Q61_cand7_friends',
        'Q61_cand3_aggressive',
        'Q61_cand4_moral',
        'Q61_cand14_competent',
        'Q61_cand15_votefor',
        'Q61_cand16_volunteer',
        'Q61_cand17_persuade',
        'Q70_mess14_imprtnt',
        'Q70_mess15_inform',
        'Q70_mess13_fair',
        'Q71_cand1_strong',
        'Q71_cand13_relate',
        'Q71_cand6_weak',
        'Q71_cand2_dishonest',
        'Q71_cand7_friends',
        'Q71_cand3_aggressive',
        'Q71_cand4_moral',
        'Q71_cand14_competent',
        'Q71_cand15_votefor',
        'Q71_cand16_volunteer',
        'Q71_cand17_persuade'
       ]]

df_all.to_csv('A1-SDO_Campaigns_All.csv', index=False)


# In[9]:


df_filter = df.loc[
    :, ['ResponseId',
        'Age',
        'Ethnicity',
        'Sex',
        'EXP_Cond',
        'EXP_Cond_HR',
        'sdo1_Pro_Trait_Dom1',
        'sdo13_Pro_Trait_Dom2',
        'sdo6_Con_Trait_Dom2',
        'sdo2_Con_Trait_Dom1',
        'sdo7_Pro_Trait_AntiEgal1',
        'sdo3_Pro_Trait_AntiEgal2',
        'sdo4_Con_Trait_AntiEgal2',
        'sdo14_Con_Trait_AntiEgal1',
        'ideol2_social',
        'ideol4_econ',
        'ideol3_self',
        'trust13_officials',
        'trust6_nocare',
        'trust2_nosay',
        'pol_interest',
        'pol_vote',
        'cand17_persuade',
        'cand16_volunteer',
        'cand15_votefor',
        'cand14_competent',
        'cand13_relate',
        'cand7_friends',
        'cand6_weak',
        'cand4_moral',
        'cand3_aggressive',
        'cand2_dishonest',
        'cand1_strong',
        'mess15_inform',
        'mess14_imprtnt',
        'mess13_fair'
       ]]

df_filter.to_csv('A2-SDO_Campaigns_filter.csv', index=False)


# ## Combining Like Variables 
# **Sorting Test Conditions**
# * Q20/Q21 - (H-SDO) & (Civil-Positive) - 
#     * Elected, strengthen border securty and defend country
#    
# * Q30/Q31 - (L-SDO) & (Civil-Positive) - 
#     * Elected, push global engagement, open country to legal migration
# * Q40/Q41 - (H-SDO) & (Civil-Negative) - 
#     * My opponent refuses to defend our country, calling for open immigration
# * Q50/Q51 - (L-SDO) & (Civil-Negative) - 
#     * Opponent refuses to push for global engagement, open country to legal migration
# * Q60/Q61 - (H-SDO) & (UnCivil) - 
#     * Opponent is a sleaze, refuses to defend country, calling for open immigration
# * Q70/Q71 - (L-SDO) & (UnCivil) - 
#     * Opponent is a sleaze who refuses global engagement, and to open immigration
# 
# 
# **Question texts**
# * **Q20/30/40/50/60/70**
#     * **14** - I think Robert Gardner raises important questions through this advertisement.
#     * **15** - I think his message is informative.
#     * **13** - This advertisement represents a fair political campaign.
# 
# * **21/31/41/51/61/71**
#     * **1** - I think Robert Gardner is a strong leader.
#     * **13** -I feel Robert Gardner is able to understand and relate to average Americans' concerns.
#     * **6** - I think Robert Gardner is a weak leader.
#     * **2** - I believe Robert Gardner is dishonest.
#     * **7** - I could become friends with Robert Gardner.
#     * **3** - I believe that Robert Gardner is an aggressive person.
#     * **4** - Robert Gardner seems to be a good, moral person.
#     * **14** - I am confident that Robert Gardner would be a competent congressional representative.
#     * **15** - I would probably vote for Robert Gardner.
#     * **16** - If asked, I would contribute time or money to Robert Gardner’s campaign.
#     * **17** - Robert Gardner’s advertisement is effective at persuading undecided voters to elect him.
