import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
from random import randint, randrange

def scrape_lyrics():
    '''
    Extract the lyrics from azlyrics.com into a dict,
    with random request intervals of 1-60 seconds*.

    Output
    ------
    df : [album_deets, song_name, lyrics]

    *You still get banned for suspicious activity :( sorry azlyrics for scaring you so much! 
    '''
    main_url = 'https://www.azlyrics.com/b/bsb.html'
    main_page = requests.get(main_url)
    soup = BeautifulSoup(main_page.content, 'html.parser')
    results = soup.find_all('div', class_='listalbum-item')

    lyrics_dict = {}
    i = 0

    # Go through all the lyric pages
    # and extract the lyrics into a dict
    for result in results:
        song_name =  result.text
        song_url = result.find('a', href=True)['href'].replace('../','')
        
        try:
            page = requests.get('https://www.azlyrics.com/{}'.format(song_url))
            lyrics_soup = BeautifulSoup(page.content, 'html.parser')
            lyrics = lyrics_soup.find_all('div', attrs={'class':None})[0].text
            album_deets = lyrics_soup.find_all('div', class_="songinalbum_title")[0].text
        except Exception as e:
            print('Skipping {song_name} as there was a {error} error, possibly due to the url: {url}'.format(song_name=song_name, error=e, url=song_url))
            continue

        lyrics_dict[i] = [album_deets, song_name, lyrics]

        i+=1
        wait_time = random.randrange(1, 60)
        print("waiting for {}s...".format(wait_time))
        time.sleep(wait_time)

    # Save the data into a file for later use.
    lyrics_df = DataFrame(lyrics_dict)
    lyrics_df_transposed = lyrics_df.T
    lyrics_df_transposed.to_csv('lyrics_data_{}.csv'.format(int(time.time())))

    return lyrics_df_transposed.rename(columns={0:'album', 1:'title', 2:'lyrics'})


