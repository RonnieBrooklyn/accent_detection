import googleapiclient.discovery
import os
import numpy as np

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../Documents/fb-mle-march-21-adebbdb5b0e5.json'


def predict_json(instances, project='fb-mle-march-21', model='accent',  version=None):

    
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def probable_accent(test_predictions):
    max_vals = []
    for i in range(len(test_predictions)):
        max_vals.append(np.argmax(test_predictions[i]['dense_3']))
    prob = max_vals.count(max(max_vals))  / len(max_vals)
    most_occuring = max(max_vals)
    return most_occuring, prob

def final_prediction(MFCCs):
    test_predictions = predict_json(MFCCs.tolist())
    accent_number, probability = probable_accent(test_predictions=test_predictions)
    return accent_number, probability