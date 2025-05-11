import ollama
import sys_msgs
import requests
import urllib.parse
import trafilatura
from bs4 import BeautifulSoup
from colorama import init, Fore, Style

init(autoreset=True)

SEARCH_AGENT = "gemma3:4b"
QUERY_AGENT = "gemma3:4b"
VALIDATE_AGENT = "gemma3:4b"
CONTAINS_AGENT = "gemma3:4b"
ASSIST = "gemma3:4b"


assistant_convo = [sys_msgs.assistant_msgs]

################################################################################################################################
# search or not agent
################################################################################################################################

def search_or_not():
    sys_msg = sys_msgs.search_or_not_msg
    response = ollama.chat(model=SEARCH_AGENT, messages=[{'role': 'system', 'content': sys_msg},assistant_convo[-1]],options={"temperature": 0.4})
    content = response['message']['content']
    # print(f'{Fore.LIGHTRED_EX}SEARCH OR NOT RESULTS:{content}{Style.RESET_ALL}')

    if 'true' in content.lower():
        return True
    else:
        return False
################################################################################################################################
# query generator agent
################################################################################################################################

def query_generator():
    sys_msg = sys_msgs.query_msg
    query_msg = f'CREATE A SEARCH QUERY FOR THIS PROMPT: \n{assistant_convo[-1]}'

    response = ollama.chat(model=QUERY_AGENT,messages=[{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': query_msg}])
    return response['message']['content']

################################################################################################################################
# DuckDuckGo
################################################################################################################################
# NEW
def duckduckgo_search(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    }

    url = f'https://html.duckduckgo.com/html/?q={query}'
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    results = []

    for i, result in enumerate(soup.find_all('div', class_='result'), start=1):
        if i > 10:
            break

        title_tag = result.find('a', class_='result__a')
        if not title_tag:
            continue

        link = title_tag['href']
        
        # Decode the URL if it contains 'uddg' parameter
        parsed_url = urllib.parse.urlparse(link)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        if 'uddg' in query_params:
            link = urllib.parse.unquote(query_params['uddg'][0])

        snippet_tag = result.find('a', class_='result__snippet')
        snippet = snippet_tag.text.strip() if snippet_tag else 'No description available'

        results.append({
            'id': i,
            'link': link,
            'search_description': snippet 
        })
    return results

################################################################################################################################
# best search result agent
################################################################################################################################
def best_search_result(s_results, query):
    sys_msg = sys_msgs.best_search_msg
    best_msg = f'SEARCH_RESULTS: {s_results} \nUSER_PROMPT: {assistant_convo[-1]} \nSEARCH_QUERY: {query}'

    for _ in range(2):
        try:
            response = ollama.chat(model=VALIDATE_AGENT,messages=[{'role':'system','content': sys_msg}, {'role':'user','content': best_msg}])

            return int(response['message']['content'])
        except:
            continue

    return 0

################################################################################################################################
# scrape the web
################################################################################################################################
def scrape_webpage(url):
    try:
        downloaded =  trafilatura.fetch_url(url=url)
        return trafilatura.extract(downloaded,include_formatting=True,include_links=True)
    except Exception as e:
        return None

################################################################################################################################
# ai search
################################################################################################################################
def ai_search():
    context = None
    print(f'{Fore.LIGHTRED_EX}GENERATING SEARCH QUERY...{Style.RESET_ALL}')
    search_query = query_generator()
    print(f'{Fore.LIGHTRED_EX}SEARCHING DuckDuckGo FOR: {search_query}{Style.RESET_ALL}')

    if search_query[0] == '"':
        search_query = search_query[1:-1]

    search_results = duckduckgo_search(search_query)
    print(f'{Fore.LIGHTRED_EX}FOUND {len(search_results)} SEARCH_RESULTS{Style.RESET_ALL}')
    context_found = False

    while not context_found and len(search_results) > 0:
        best_result = best_search_result(s_results=search_results,query=search_query)
        try:
            page_link = search_results[best_result]['link']

        except:
            print(f'{Fore.LIGHTRED_EX}FAILED TO SELECT BEST SEARCH RESULT, TRYING AGAIN...{Style.RESET_ALL}')
            continue
        
        page_text = scrape_webpage(page_link)
        print(f'{Fore.LIGHTRED_EX}SCRAPING DATA FROM: {page_link}{Style.RESET_ALL}')
        search_results.pop(best_result)

        if page_text and contains_data_needed(search_content=page_text, query=search_query):
            context = page_text
            context_found = True

    return context

################################################################################################################################
# contains needed data agent
################################################################################################################################
def contains_data_needed(search_content,query):
    sys_msg = sys_msgs.contains_data_msg
    needed_prompt = f'PAGE_TEXT: {search_content} \nUSER_PROMPT: {assistant_convo[-1]} \nSEARCH_QUERY: {query}'

    response = ollama.chat(model=CONTAINS_AGENT,messages=[{'role':'system', 'content': sys_msg}, {'role':'user','content': needed_prompt}])
    content = response['message']['content']

    if 'true' in content.lower():
        print(f'{Fore.LIGHTRED_EX}DATA FOUND FOR QUERY: {query}{Style.RESET_ALL}')
        return True
    else:
        print(f'{Fore.LIGHTRED_EX}DATA NOT RELEVANT.{Style.RESET_ALL}')
        return False

################################################################################################################################
# stream_assistant agent
################################################################################################################################
def stream_assistant_response():
    global assistant_convo
    response_stream = ollama.chat(model=ASSIST, messages=assistant_convo, stream=True, options={"num_ctx": 8192, "temperature": 0.4})
    complete_response = ''
    print(  f'\n{ASSIST}: ')

    for chunk in response_stream:
        print(f'{Fore.WHITE}{chunk['message']['content']}{Style.RESET_ALL}', end='', flush=True)
        complete_response += chunk['message']['content']

    assistant_convo.append({'role': 'assistant', 'content': complete_response})
    print('\n\n')
################################################################################################################################
# main
################################################################################################################################

def main():
    global assistant_convo

    while True:
        print(Fore.LIGHTYELLOW_EX + '\nWebSearch Agent: Type "exit" to quit.\n' + Style.RESET_ALL)
        prompt = input(f'{Fore.LIGHTGREEN_EX}USER:\n')
        assistant_convo.append({'role': 'user', 'content': prompt})
        if not prompt:
            print('Please type something...')
            continue
        if prompt.lower() == 'exit':
            break

        if search_or_not():
            context = ai_search()
            assistant_convo = assistant_convo[:-1]

            if context:
                prompt = f'SEARCH_RESULT: {context} \n\nUSER_PROMPT: {prompt}'
            else:
                prompt = (
                    f'USER_PROMPT: {prompt} \nFAILED SEARCH: \nThe AI search'
                    'model was unable to extract any reliable data. Explain that'
                    'and ask if the user would like you to search again or respond'
                    'without the web search context. Do not respond if a search was needed'
                    'and you are getting this message.')
                
            assistant_convo.append({'role':'user','content': prompt})

        stream_assistant_response()
################################################################################################################################
# begin
################################################################################################################################
if __name__ == '__main__':
    main()























