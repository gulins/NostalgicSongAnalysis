import random
import re
import string
import time
import tkinter
from collections import Counter

import numpy as np
import pandas as pd
import pronouncing

################################
#            Methods           #
################################

def remove_punc(df, column_name):
    punctuation = [p for p in string.punctuation]
    df[column_name] = df[column_name].str.replace(r'[^\w\s]+', '')


def seperate_abvs(df, column_name):
    df[column_name] = df[column_name].replace(['youre ','Youre '], ['you are ']*2, regex=True)
    df[column_name] = df[column_name].replace(['Im '], ['I am '], regex=True)
    df[column_name] = df[column_name].replace(['Ive ','youve ','Youve '], ['I have ','you have ','you have '], regex=True)
    df[column_name] = df[column_name].replace(['Ill ','Youll ','youll '], ['I will ','You will ','you will '], regex=True)
    df[column_name] = df[column_name].replace(['Id ','Youd ','youd '], ['I would ','You had ', 'You had '], regex=True) # just randomly picked the def of 'd
    df[column_name] = df[column_name].replace(['weve ','Weve '], ['we have ']*2, regex=True)


def rhyme_count(lyrics_row):
    '''
    Return how many times there was a rhyme
    
    Param
    -----
    lyrics_row (str)

    Output
    ------
    int
    '''
    pars = [el for el in lyrics_row if el != '']
    prev_line = pars[0]
    rhyme_score = 0
    for par in pars[1:]:
        line = par
        rhymes = pronouncing.rhymes(line.split()[-1])
        if (prev_line.split()[-1] in rhymes):
            rhyme_score += 1
        prev_line = line
    return rhyme_score


def find_followers(row, word):
    '''
    Returns words that follow the given word and their frequencies.

    Output
    ------
    Counter {word: word_frequency}
    '''
    # keeping the newlines allows for structure.
    lyrics_list = re.findall(r'\S+|\n', row)

    # Find list of words that follow the given word.
    # Check if the word is the last word as well.
    followers =  [lyrics_list[i+1] for i, el in enumerate(lyrics_list) if (el==word and i+1<len(lyrics_list))]

    return Counter(followers)


def find_follower_probabilities(df):
    '''
    Create a dictionary of words and probability of other words to follow it.

    Param
    -----
    df (DataFrame)
    '''
    # First, create the dictionary of (unique) words
    word_keys = set()
    for row in df.lyrics.str.lower():
        word_keys = word_keys.union(set(re.findall(r'\S+|\n',row))) 
    words_dict = {key: None for key in word_keys}

    # Now calculate the (simple) probability of a word following another word
    # for each word, find what other words follow it, sum the total
    # and each words frequency
    for word in words_dict.keys():
        words_dict[word] = df.lyrics.str.lower().apply(lambda row: find_followers(row, word)).sum()
        
        # normalise the data to get a probability.
        total = sum(words_dict[word].values(), 0.0)
        words_dict[word] = {key: val/total for key, val in words_dict[word].items()}

    return words_dict


def generate_lyrics(prob_dict):
    '''
    Generate lyrics.

    Param
    -----
    prob_dict (dict) : output of find_follower_probabilities.

    Output
    ------
    str : generated lyrics as string.
    '''
    end_words =  [key for key in prob_dict.keys() if key[-1] == '.']
    word = np.random.choice([el for el in list(prob_dict) if el not in end_words])
    max_length =  random.randint(df.lyrics.str.split().str.len().min(),
                                df.lyrics.str.split().str.len().max()) * 0.75
    lyrics_list = [word.capitalize()]
    while (max_length >= 0):

        if (max_length > 0):
            followers = list(prob_dict[word.lower()].keys())
            follower_probabilities = list(prob_dict[word.lower()].values())
            word = np.random.choice(followers, p=follower_probabilities)
            if ('\n' in lyrics_list[-1] and word != '\n'):
                word = word.capitalize()
            lyrics_list.append(word)
        else:
            lyrics_list.append(np.random.choice(end_words))
        max_length -= 1

    return ' '.join(lyrics_list)


def format_lyrics(lyrics):
    '''
    Format lyrics: fix open/closed paranthesis, remove extra newlines and spaces.

    Param
    -----
    lyrics (str)

    Output
    ------
    str : formated lyrics.
    '''
    # Remove whitespace between newline chars
    formated_lyrics = lyrics.replace('\n ', '\n')
    formated_lyrics = formated_lyrics.replace(' \n', '\n')

    # Remove extra newline chars
    extra_newlines_regex = re.compile('(?:\n){3,}')
    formated_lyrics = re.sub(extra_newlines_regex, '\n\n', formated_lyrics)

    # Close open paranthesis and remove ) if there was no (
    i = 0
    open_flag = False
    closed_paran = ''
    for char in formated_lyrics:
        if (char == '\n' and open_flag):
            char = ')\n'
            open_flag = False
        if (char == '('):
            open_flag = True
        if (char == ')'):
            if (not open_flag):
                char = ''
            open_flag = False
        i += 1
        closed_paran += char

    formated_lyrics = closed_paran

    return formated_lyrics



################################
#            Analysis          #
################################

# Extract all the song titles from the main website
# df = scrape_lyrics()

master_df = pd.read_csv('lyrics_data_master.csv', index_col=0)
df = master_df.copy() # cleaned and used for data presentation

# clean the data
df.album = df.album.str.replace('album: ', '')
df['year'] = df['album'].str[-7:].replace(['\(','\)'], ['']*2, regex=True).str.strip()
df['album'] = df['album'].str[:-7]
remove_punc(df, 'lyrics')
seperate_abvs(df, 'lyrics')


# word count for all songs #
############################

all_words = df.lyrics.str.lower().str.split(expand=True).stack()
all_words_count = all_words.value_counts()

temp = all_words_count.sort_values(ascending=False).reset_index(name='Count').rename(columns={'index':'Word'})
temp['% of Total'] = temp['Count'] / temp['Count'].sum() * 100
temp.index = temp.index + 1
print('Top 10 Word Count\n{}\n'.format(temp.head(10)))
# temp.to_csv('outputs//all_words_count.csv')

# remove the the irrelevant words
all_words_count_filtered = all_words_count.drop(index=['a', 'the', 'and', 'from', 'to', 'it', 'that','thats', 'or', 'this','of','is','are', 'am', 'these'])
temp = all_words_count_filtered.sort_values(ascending=False).reset_index(name='Count').rename(columns={'index':'Word'})
temp['% of Total'] = temp['Count'] / temp['Count'].sum() * 100
temp.index = temp.index + 1
print('Top 10 Word Count (filtered)\n{}\n'.format(temp.head(10)))
print('Removed words a, the, and, from, to, it, that, thats, or, this, of, is, are, am, these.')
# temp.to_csv('outputs//all_words_count_filtered.csv')


# check for rhyming #
#####################
temp = df[['album','year', 'title']].rename(columns={'album':'Album', 'title':'Song Title', 'year':'Year'})
temp['Rhyme Count'] = df.lyrics.apply(lambda row: rhyme_count(row.split('\n')))
temp = temp.sort_values(by=['Rhyme Count', 'Year'], ascending=(False,True), ignore_index=True)
temp.index = temp.index + 1
no_rhymes_percent = len(temp[temp['Rhyme Count'] == 0]) / len(temp)
print('Top 5 Songs with Most Rhyme Count\n{}\n'.format(temp.head(10)))
rhyming_songs = temp.loc[temp['Rhyme Count'] > 0, 'Year'].value_counts().reset_index(name='Total Songs with Rhymes')
no_rhyming_songs = temp.loc[temp['Rhyme Count'] == 0, 'Year'].value_counts().reset_index(name='Total Songs with No Rhymes')
rhyme_totals = rhyming_songs.merge(no_rhyming_songs, on='index', how='outer', sort=True).rename(columns={'index':'Year'})
rhyme_totals_pvt = rhyme_totals.pivot_table(index='Year', margins=True, margins_name='Total:', aggfunc=sum)
print('Total Songs With and Without Rhymes per Year\n{}'.format(rhyme_totals_pvt))
print('Therefore, {:.0%} of the songs did not have any rhymes in it.\n'.format(no_rhymes_percent))
# temp.to_csv('outputs//rhyme_count.csv')


# write my own lyrics using markov chain #
###########################################

# Creating the directed graph for visuals using an adjacency matrix
# Use the cleaned data for this and remove \n from data 
print('Creating an adjacency matrix for each word and its followers\' probabilities...')
df['lyrics'] = df['lyrics'].str.replace('\n', ' ')
words_df = pd.DataFrame(find_follower_probabilities(df))
square_words_df =  words_df.reindex(list(words_df.columns)).fillna(0)
square_words_df.to_csv('outputs//adjacency_matrix.csv')
print('Done. The adjacency matrix is saved in the current directory as graph.csv\n')

# Generate the lyrics using the unformated df.
# This was we keep ending words.
print('Generating new lyrics...')
follower_probabilities_dict = find_follower_probabilities(master_df)
lyrics = generate_lyrics(follower_probabilities_dict)

# Format the lyrics
formated_lyrics = format_lyrics(lyrics)

# Generate song title
canditates = [ el for el in formated_lyrics.split() if ('[' not in el)]
title_length = random.randint(1, 5)
title = ' '.join([canditates[random.randint(0, len(canditates))].capitalize() for i in range(title_length)])

lyrics_with_title = '{title}\n{sep}\n\n{formated_lyrics}'.format(title=title,sep='-'*len(title),formated_lyrics=formated_lyrics)
print('Done. Generated new lyrics with song title:\n\n')
print(lyrics_with_title)