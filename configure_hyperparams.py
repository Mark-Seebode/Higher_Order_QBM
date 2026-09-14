import wandb
import argparse
import pickle


name = 'fully visi 1o Z ZZ ZZZ UCI'
project = ""
entity = ""

sweep_configuration = {'name': name,
                       'description': 'looking for best params?',
                       'project': project, 'entity': entity,
                       'method': 'bayes',
                       'metric': {'goal': 'maximize', 'name': 'acc'},
                       'parameters': {'batch_size': {'values': [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8]},#, 2**9]},


                                      'learning_rate': {'max': 0.5, 'min': 0.0005},
                                      'chebyshev_degree': {'values': [2, 4, 6, 8, 10]},
                                      'beta': {'max': -1.0, 'min': -6.0},
                                      },
                       'early_terminate': {'type': 'hyperband', 'min_iter': 4}
                       }


if __name__ == '__main__':
    with open("wandb_key.txt", "r") as f:
       key = f.read().strip()
    wandb.login(key=key)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project, entity=entity)
    print(list(range(10, 401, 10)))
    #initialize_counter()




