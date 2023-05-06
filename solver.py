import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from config import images_dir


class Solver:
    def __init__(self, model, data, learning_rate, batch_size, num_epochs, checkpoint_name=None, lr_decay=0,
                 verbose=True, print_every=100):
        self.model = model
        self.X_train = data['X_train']
        self.Y_train = data['Y_train']
        self.X_val = data['X_test']
        self.Y_val = data['Y_test']
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_every = print_every
        self.lr_decay = lr_decay
        self.checkpoint_name = checkpoint_name

        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.epoch = 0

    def train(self):
        num_train = self.X_train.shape[1]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        print(num_iterations)
        for i in range(num_iterations):
            self.step()
            # print training loss
            if self.verbose and i % self.print_every == 0:
                print("(Iteration %d / %d) loss: %f, lr: %f" % (
                    i + 1, num_iterations, self.loss_history[-1], self.learning_rate))

            epoch_end = (i + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                self.learning_rate *= self.lr_decay

            first_it = i == 0
            last_it = i == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc, _ = self.check_accuracy(self.X_train, self.Y_train)
                val_acc, _ = self.check_accuracy(self.X_val, self.Y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %lf, val acc: %f' %
                          (self.epoch, self.num_epochs, train_acc, val_acc))
        plt.plot(self.loss_history)
        plt.title('Cross Entropy Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(images_dir, 'loss.png'))
        plt.show()
        p1, = plt.plot(self.train_acc_history)
        p2, = plt.plot(self.val_acc_history)
        plt.legend([p1, p2], ['train acc', 'val acc'], loc='lower right')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(images_dir, 'accuracy.png'))
        plt.show()

    def step(self):
        num_train = self.X_train.shape[1]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[:, batch_mask]
        Y_batch = self.Y_train[batch_mask]
        # print(f'step--num_train: {num_train}, X: {X_batch.shape}, Y: {Y_batch.shape}')

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, Y_batch)
        self.loss_history.append(loss)
        # parameter update
        params = self.model.parameters
        L = len(params) // 2
        for i in range(1, L + 1):
            params['W' + str(i)] = params['W' + str(i)] - self.learning_rate * grads['dW' + str(i)]
            params['b' + str(i)] = params['b' + str(i)] - self.learning_rate * grads['db' + str(i)]

    def check_accuracy(self, X, Y, batch_size=100):
        N = X.shape[1]
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        Y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[:, start:end])

            Y_pred.append(np.argmax(scores, axis=0))
        Y_pred = np.hstack(Y_pred)
        acc = np.mean(Y_pred == Y)
        return acc, Y_pred

    def save_model(self):
        saved_model = {
            'model': self.model,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'lr_decay': self.lr_decay,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history
        }
        file_name = f'{self.checkpoint_name}.pkl'
        if self.verbose:
            print(f'Saving model to {file_name}')
        with open(file_name, 'wb') as f:
            pickle.dump(saved_model, f)

    @classmethod
    def load_model(cls, file_name, data, batch_size=0, num_epochs=0, checkpoint_name=None, verbose=True,
                   print_every=100):
        with open(file_name, 'rb') as f:
            m = pickle.load(f)
        solver = Solver(m['model'], data, m['learning_rate'], batch_size, num_epochs, checkpoint_name,
                        m['lr_decay'], verbose, print_every)
        solver.loss_history = m['loss_history']
        solver.train_acc_history = m['train_acc_history']
        solver.val_acc_history = m['val_acc_history']
        return solver
