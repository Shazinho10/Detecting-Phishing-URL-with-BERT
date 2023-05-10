#extracting the data from json files
import json
def data_extraction(path_to_json):

  f = open(path_to_json)
  file = json.load(f)

  urls = [i for i in file.keys()]

  labels = [file[urls[i]].keys() for i in range(len(urls))]
  labels = [list(val)[0] for val in labels]

  text = [file[urls[i]].values() for i in range(len(urls))]
  text = [list(val)[0] for val in text]

  return text, labels, urls