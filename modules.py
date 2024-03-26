from sklearn.metrics import accuracy_score
import time 
from numpy import reshape
from sklearn.model_selection import train_test_split

def print_metrics(predicts, models_names, y_test, transformer_name='', message = 'Without transform'):
    accuracy = {}
    for name in models_names:
            predict = predicts[name]
            print(f'Model {name} results {message} {transformer_name}:')
            accuracy[name] = accuracy_score(y_true=y_test, y_pred=predict)
            print(accuracy[name], '\n')
            # print(_score(y_true=y_test, y_pred=predicts[name]))
    return accuracy

def print_time_report(times, name):
    if name == 'PCA':
        times_train = times[name + '_train']
        times_test = times[name + '_test']
        print(f'Алгоритм {name}:')
        print(f'X_train: {times_train} сек')
        print(f'X_test: {times_test} сек')
    elif name == 'TSNE':
        times_train = times[name]
        print(f'Алгоритм {name}:')
        print(f'X_train: {times_train} сек')
        




def fit_predict(models, eval_data):
    predicts = {}
    for name, model in models.items():
        model.fit(eval_data['X_train'], eval_data['y_train'])
        models[name] = model
        predicts[name] = model.predict(eval_data['X_test'])

    return models, predicts



def transform_fit_predict(models, data, transformers):
    times = {}
    predicts = {}
    models_new = {}
    for name, transformer in transformers.items():
        if name == 'PCA':
            X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size=0.25, random_state=42, stratify=data['y'])

            # Вычислим время, необходимое для преобразования Тренировочной выборки
            start = time.time()
            transformer.fit(X_train)
            X_train_ = transformer.transform(X_train)
            times[name + '_train'] = time.time() - start

            # Вычислим время, необходимое для преобразования Тестовой выборки
            start = time.time()
            X_test_ = transformer.transform(X_test)
            times[name + '_test'] = time.time() - start
        elif name == 'TSNE':

            # Вычислим время, необходимое для преобразования Тренировочной выборки
            start = time.time()
            X_ = transformer.fit_transform(data['X'])
            times[name] = time.time() - start

            X_train_, X_test_, y_train, y_test = train_test_split(X_, data['y'], test_size=0.25, random_state=42, stratify=data['y'])


        eval_data_ = {
            'X_train': X_train_,
            'X_test': X_test_,
            'y_train': y_train,
            'y_test': y_test
        }
        
        models_, predict = fit_predict(models, eval_data_)
        models_new[name] = models_
        predicts[name] = predict
    return models_new, transformers, times, predicts
    
    

