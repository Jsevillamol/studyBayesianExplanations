from flask import Flask
from flask import request
from flask import render_template
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

import os
from pgmpy.readwrite import BIFReader
import numpy as np

def load_network(network_name):
  url = f"https://www.bnlearn.com/bnrepository/{network_name}/{network_name}.bif.gz"
  os.system(f"wget {url} -q")
  fn = f"{network_name}.bif.gz"
  os.system(f"gzip -qd -f {fn} -q")
  fn = f"{network_name}.bif"
  reader = BIFReader(fn)
  os.system(f"rm {fn}")
  model = reader.get_model()
  
  return model

import requests
def load_network(network_name):
  url = f"https://www.bnlearn.com/bnrepository/{network_name}/{network_name}.bif.gz"
  r = requests.get(url, allow_redirects=True)
  fn = f"{network_name}.bif.gz"
  open(fn, 'wb').write(r.content)
  os.system(f"gzip -qd -f {fn} -q")
  fn = f"{network_name}.bif"
  reader = BIFReader(fn)
  os.system(f"rm {fn}")
  model = reader.get_model()
  model.states = reader.get_states()
  
  return model

random_evidence = lambda model, evidence_nodes : \
   {node : np.random.choice(list(model.states[node])) \
    for node in evidence_nodes if np.random.rand() < 0.5}

@app.route('/')
def main_page():
  # prepare model and evidence
  model = load_network('asia')
  target = np.random.choice(model.nodes)
  evidence_nodes = [node for node in model.nodes if node != target]
  evidence = random_evidence(model, evidence_nodes)
  
  # Prepare HTML components
  evidence_explanation = [f"{node} = {state}" for node, state in evidence.items()]
  evidence_html = \
      render_template('evidence.html', 
                      evidence_explanation = evidence_explanation)
  visualization_html = \
      render_template('visualization.html',
                      html_element_id = 'explanation_control_mistake',
                      bn_model = model,
                      squeeze_fn = np.squeeze, # Yes I know this is very hacky
                      )
  survey_html = render_template('survey.html')
                      
  return render_template('main.html',
                         evidence_html = evidence_html,
                         visualization_html = visualization_html,
                         survey_html = survey_html,
                         )
  

if __name__ == '__main__':
    app.run()