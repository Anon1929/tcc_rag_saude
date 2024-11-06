
import requests
import pandas as pd


def generate_questions(context_texts):
    questions = []
    model_url = "http://localhost:11434/api/generate"  

    for context in context_texts:
        prompt = f"Baseado no seguinte texto, gere uma pergunta:\n\n{context}\n\nPergunta:"
        
        payload = {
            "model": "llama3.1",
            "prompt": prompt,
            "max_tokens": 250,  
            "stream": False
        }
        
        try:
            response = requests.post(model_url, json=payload)
            response.raise_for_status()  
            question = response.json().get("response")
            questions.append(question)
        except Exception as e:
            print(f"Error generating question: {e}")
    
    return questions

if __name__ == "__main__":

    df = pd.read_csv("ContextosSelecionados.csv", sep="|")


    text_list = df["text"].tolist()
        
    questions = generate_questions(text_list)
    

    dfDataset = pd.DataFrame({
        "question": questions,
        "ground_truths": text_list   
    })

    dfDataset.to_csv("DataSetPerguntas.csv",sep = '|', index=True)