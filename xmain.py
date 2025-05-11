import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta, timezone
from queue import Queue
from time import sleep
from sys import platform
from ollama import chat # Added for Ollama integration
from speak import speak # Import the speak function
from colorama import init, Fore, Style

init(autoreset=True)

# Import the search_agent module
import search_agent

WAKE_WORD = "TOPSY"
MODEL_OLLAMA = 'gemma3:4b' # Configuration from test_chat.py
SYS_MSG_OLLAMA = ( # Configuration from test_chat.py
'''System_instructions:
  role: "AI Voice Assistant"
  behavior:
    - "Respond as an expert problem solver specialized in voice interactions."
    - "Provide concise, logical, and helpful answers—avoid verbosity."
    - "Act strictly as a voice assistant, not as a general-purpose chatbot."
    - "Use clear reasoning and deductive logic to address user questions."
    - "If information is outside training scope, inform the user honestly."
  constraints:
    - "No improvisation or fabrication when knowledge is lacking."
    - "Responses must be optimized for spoken clarity and brevity."

'''
)
EXIT_PHRASE = "goodbye topsy" # Phrase to exit the assistant

# Function to generate responses using Ollama (adapted from test_chat.py)
def generate_ollama_response(convo_messages):
    """
    Sends a conversation to the Ollama model and streams the response.
    Args:
        convo_messages (list): A list of message dictionaries for the Ollama API.
    Returns:
        str: The full text of the assistant's response.
    """
    try:
        stream = chat(model=MODEL_OLLAMA, messages=convo_messages, stream=True)
        full_response_text = ''
        print('ASSISTANT:')
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            full_response_text += content
        print('\n\n') # Ensure separation after assistant's full response
        return full_response_text
    except Exception as e:
        print(f"\n[Ollama Error] Failed to get response: {e}")
        return "Sorry, I encountered an error trying to respond."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Whisper model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Use non-English model (if available for chosen size)")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect sound", type=int)
    parser.add_argument("--record_timeout", default=10, # User's original default
                        help="How long to record audio for a single phrase (seconds) by SpeechRecognition", type=float)
    parser.add_argument("--phrase_timeout", default=15, # User's original default
                        help="How much silence before a voice command is considered over (seconds)", type=float)
    args = parser.parse_args()

    # --- Initialization ---
    data_queue = Queue()    # Queue for raw audio data from microphone
    phrase_bytes = bytes()  # Buffer for raw audio data of the current phrase being captured
    triggered = False       # True if wake word has been detected and assistant is listening for a command
    last_phrase_time = None # Timestamp of the last audio data received for the current phrase
    full_transcript = []    # Holds the words of the command being spoken post-wake-word

    # Initialize chat history for Ollama with the system message
    chat_history = [{'role': 'system', 'content': SYS_MSG_OLLAMA}]

    # SpeechRecognition setup
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False # Use a consistent energy threshold

    # Load Whisper model
    whisper_model_name = args.model
    if not args.non_english and args.model != "large": # Append .en for English-specific models
        whisper_model_name += ".en"
    
    print("🔁 Initializing Whisper model... This might take a moment.")
    try:
        audio_model = whisper.load_model(whisper_model_name)
        print(f"✅ Whisper model '{whisper_model_name}' loaded.")
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        print("Please ensure you have the correct model name and dependencies installed.")
        return

    # Microphone setup
    print("🎤 Initializing microphone...")
    try:
        source = sr.Microphone(sample_rate=16000) # Whisper prefers 16kHz
    except Exception as e:
        print(f"❌ Error initializing microphone: {e}")
        print("Please ensure a microphone is connected and configured.")
        return

    print("🎧 Calibrating ambient noise... Please be silent.")
    with source:
        recorder.adjust_for_ambient_noise(source, duration=1) # Calibrate for 1 second
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"✅ Ready. Say '{WAKE_WORD}' to begin. To exit, say '{EXIT_PHRASE}' after the wake word.")

    def record_callback(_, audio: sr.AudioData):
        """
        Callback function called by SpeechRecognition when a phrase is detected.
        Adds raw audio data to the queue.
        """
        data = audio.get_raw_data()
        data_queue.put(data)

    # Start listening in the background
    # This returns a function to stop listening, which can be used for cleanup
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    # --- Main Loop ---
    try:
        while True:
            now = datetime.now(timezone.utc)

            # 1. Process new audio data from the queue
            if not data_queue.empty():
                audio_chunk = b''.join(list(data_queue.queue)) # Consume all items in queue
                data_queue.queue.clear()
                phrase_bytes += audio_chunk # Append to current phrase buffer
                last_phrase_time = now      # Update time of last audio activity

                # Transcribe the current accumulated audio if there's content
                if phrase_bytes:
                    audio_np = np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    if audio_np.shape[0] > 0: # Ensure there's actual audio data to process
                        result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                        text_from_whisper = result['text'].strip()

                        if not triggered: # If wake word not yet detected
                            if WAKE_WORD.lower() in text_from_whisper.lower():
                                os.system('cls' if os.name == 'nt' else 'clear')
                                print(f"🟢 Wake word '{WAKE_WORD}' detected!")
                                speak("Listening for your command...") # Voice prompt
                                triggered = True
                                phrase_bytes = bytes()  # Clear buffer to capture only the command
                                full_transcript = []    # Reset transcript for the command
                                last_phrase_time = now  # Reset timer for the command phrase itself
                                print("🎙️ Listening for your command...\n")
                        elif triggered: # If wake word IS active, we are capturing the command
                            if text_from_whisper: # If Whisper transcribed something for the command
                                full_transcript = [text_from_whisper] # Update with the latest full transcription
                            
                            # Display current state of command recognition
                            os.system('cls' if os.name == 'nt' else 'clear')
                            print("🎙️ Listening for your command...\n")
                            if full_transcript:
                                print(" ".join(full_transcript))
                            else:
                                print("...") # Indicate listening but no speech detected yet for command
            
            # 2. Check for command timeout (if wake word was triggered)
            if triggered and last_phrase_time and (now - last_phrase_time > timedelta(seconds=args.phrase_timeout)):
                user_command = " ".join(full_transcript).strip()
                
                os.system('cls' if os.name == 'nt' else 'clear')
                current_display_command = ' '.join(full_transcript) if full_transcript else "..."
                print(f"🎙️ Listening for your command...\n{current_display_command}")
                
                if user_command:
                    print(f"\n⏹️ Silence detected. Processing command: \"{user_command}\"")
                    if user_command.lower() == EXIT_PHRASE.lower():
                        print(f"\n🛑 Exit phrase '{EXIT_PHRASE}' detected. Shutting down.")
                        speak("Goodbye!") # Voice prompt on exit
                        break # Exit the main loop
                    
                    # Prepare a temporary chat history for search_agent to determine if a search is needed
                    # This will allow search_agent's search_or_not and ai_search to use the user's command
                    original_assistant_convo = list(search_agent.assistant_convo) # Store original to restore later
                    search_agent.assistant_convo.append({'role': 'user', 'content': user_command})
                    
                    context = None
                    try:
                        if search_agent.search_or_not():
                            print(f"{Fore.LIGHTRED_EX}PERFORMING WEB SEARCH...{Style.RESET_ALL}")
                            context = search_agent.ai_search()
                            if context:
                                prompt_for_ollama = f"SEARCH_RESULT: {context} \n\nUSER_PROMPT: {user_command}"
                                chat_history.append({'role': 'user', 'content': prompt_for_ollama})
                            else:
                                print("\n🤷 Web search failed to retrieve relevant information.")
                                speak("I couldn't find relevant information online. I will try to respond without it.")
                                chat_history.append({'role': 'user', 'content': user_command}) # Proceed without context
                        else:
                            chat_history.append({'role': 'user', 'content': user_command})
                    except Exception as e:
                        print(f"\n[Search Agent Error] An error occurred during web search: {e}")
                        speak("I encountered an issue with the web search. I will try to respond without it.")
                        chat_history.append({'role': 'user', 'content': user_command}) # Fallback to no search if error
                    finally:
                        # Restore search_agent's conversation history to its original state
                        search_agent.assistant_convo = original_assistant_convo
                        
                    assistant_response = generate_ollama_response(chat_history)
                    if assistant_response: # Add to history only if a response was generated
                        chat_history.append({'role': 'assistant', 'content': assistant_response})
                        speak(assistant_response) # Speak the assistant's response
                else:
                    print("\n🤷 No command content spoken after wake word.")
                    speak("Sorry, I didn't catch that. Please say your command again after the wake word.") # Voice prompt

                # Reset for next wake word cycle (unless we are exiting)
                triggered = False
                full_transcript = []
                phrase_bytes = bytes()    # Clear audio buffer for next round
                last_phrase_time = None   # Reset last phrase time to wait for new audio
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"✅ Ready. Say '{WAKE_WORD}' to begin. To exit, say '{EXIT_PHRASE}' after the wake word.")
                # Clear queue of any audio that might have arrived during Ollama processing
                while not data_queue.empty(): 
                    try:
                        data_queue.get_nowait()
                    except Exception:
                        break

            # 3. Check for general timeout (if not triggered, to clear long silent audio accumulation)
            elif not triggered and last_phrase_time and (now - last_phrase_time > timedelta(seconds=args.phrase_timeout * 2)):
                if len(phrase_bytes) > 0:
                    phrase_bytes = bytes()
                    full_transcript = [] 
                last_phrase_time = None 

            sleep(0.15) 

    except KeyboardInterrupt:
        print("\n🛑 Exiting voice assistant via KeyboardInterrupt.")
        speak("Exiting. Goodbye!") # Voice prompt on keyboard interrupt
    except Exception as e:
        print(f"\n⚠️ An unexpected error occurred: {e}")
        speak("An unexpected error occurred.") # Voice prompt on error
        import traceback
        traceback.print_exc() 
    finally:
        print("Cleaning up...")
        if 'stop_listening' in locals() and stop_listening:
            stop_listening(wait_for_stop=False) 
        print("Goodbye!")

if __name__ == "__main__":
    main()