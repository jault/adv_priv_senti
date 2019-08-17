import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import DataParallel, CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler

from networks import BasicCNN, BasicAutoEncoder
import iemocap

DATA_PATH = os.path.join('.', 'IEMOCAP')
TASK_CLASSES = len(iemocap.emo_int)
PRIVACY_CLASSES = 10

EPOCHS = 2
BATCH_SIZE = 32

LAMBDA = 0.6    # < 1.0 favors task perf. over privacy perf.

# Indices for labels returned by the data manager
TASK_LABEL_IDX = 1
PRIVACY_LABEL_IDX = 2


# TODO tune parameters/networks, budget ensembling, convert spectrograms back to audio
def main():
    print('Loading dataset..')
    dataset = iemocap.IemocapAudio(DATA_PATH)
    #speaker_independent(dataset, degrade=None)  # Determine prior performance of task classification model
    print('Learning degradation function')
    dataset.print_spectrograms(affix='og', degrade=None)
    degrade = speaker_dependent(dataset)
    torch.save(degrade.state_dict(), "degrade_model.pt")
    dataset.print_spectrograms(affix='dgfn', degrade=degrade)
    print('Testing degradation function')
    speaker_independent(dataset, degrade)


def speaker_independent(dataset, degrade):
    """ To show our model meets benchmark performance for emotion recognition we need to use leave-one-speaker out cross
     validation. Otherwise we don't learn emotion in general, but emotion as portrayed by each speaker."""
    track_perf = []
    for i in range(len(dataset.subject_indices)):
        # Grab the subject's sample indices saved in the dataset object
        subject_start = dataset.subject_indices[i - 1] if i != 0 else 0
        subject_end = dataset.subject_indices[i]
        print('\n----Subject', i+1, 'start', subject_start, 'end', subject_end)
        dataset.distribution(subject_start, subject_end)

        # Test on subject and train on all remaining data
        train_idxs = list(range(0, subject_start)) + list(range(subject_end, len(dataset.data)))
        test_idxs = list(range(subject_start, subject_end))

        train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idxs), batch_size=BATCH_SIZE)
        test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idxs), batch_size=BATCH_SIZE)

        model = BasicCNN(TASK_CLASSES).cuda()

        train_task_classifier(model, train_loader, test_loader, degrade=degrade)
        val_perf = check_accuracy(model, test_loader, TASK_LABEL_IDX, degrade=degrade)
        track_perf.append(val_perf)
        print('Speaker', i + 1, 'Val Acc', val_perf)

    print('Finished leave-one-speaker out for', len(dataset.subject_indices), 'speakers')
    print(track_perf)
    print('Mean', np.mean(track_perf), 'Std', np.std(track_perf))


def speaker_dependent(dataset):
    """ Use adversarial training to learn a degradation function to protect subject identity privacy while preserving
    target task recognition. Z. Wu, 2018, 'Towards privacy-preserving visual recognition via adversarial training.'

    We can't use speaker independent cross validation when learning a degradation function because the privacy model
    can't be defined in that case. Here we select a random 20% for validation."""
    train_size = int(0.8 * len(dataset))
    indices = np.arange(len(dataset))

    # Use the same validation set each run
    random.seed(2019)
    random.shuffle(indices)
    random.seed()

    train_idxs, test_idxs = indices[:train_size], indices[train_size:]

    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idxs), batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idxs), batch_size=BATCH_SIZE)

    task_model = DataParallel(BasicCNN(TASK_CLASSES)).cuda()
    privacy_model = DataParallel(BasicCNN(PRIVACY_CLASSES)).cuda()
    degrade_model = DataParallel(BasicAutoEncoder()).cuda()

    # Pre-train classifiers without degradation
    train_task_classifier(task_model, train_loader, test_loader, degrade=None)
    train_privacy_classifier(privacy_model, train_loader, test_loader, degrade=None)
    for epoch in range(EPOCHS):
        train_degradation_function(degrade_model, task_model, privacy_model, train_loader, test_loader)
        dataset.print_spectrograms(affix='dg'+str(epoch+1), degrade=degrade_model)
        train_task_classifier(task_model, train_loader, test_loader, degrade_model)
        reset = train_privacy_classifier(privacy_model, train_loader, test_loader, degrade_model)
        if reset:
            print('Resetting privacy model')
            privacy_model = DataParallel(BasicCNN(PRIVACY_CLASSES)).cuda()
            finished = train_privacy_classifier(privacy_model, train_loader, test_loader, degrade_model)
            if finished:
                # No longer able improve performance of the subject identity detector, we have a strong degradation fn.
                return degrade_model
    return degrade_model


def train_degradation_function(degrade_model, task_model, privacy_model, train_loader, test_loader):
    task_model.eval(), privacy_model.eval()
    degrade_optimizer = Adam(degrade_model.parameters(), lr=0.0005)
    for epoch in range(EPOCHS):
        batch_loss = []
        for batch, item in enumerate(train_loader):
            inputs, labels, subjects = item[0].cuda(), item[1].cuda(), item[2].cuda()

            degraded = degrade_model(inputs)
            task_outputs, privacy_outputs = task_model(degraded), privacy_model(degraded)
            task_loss, privacy_loss = F.cross_entropy(task_outputs, labels), F.cross_entropy(privacy_outputs, subjects)

            # TODO Lambda should be adjusted down as degrade improves somehow to preserve a desired task accuracy
            # Maximize task perf., minimize privacy perf. and produce audio-like output
            degrade_loss = (task_loss / (LAMBDA * privacy_loss))
            batch_loss.append(degrade_loss.item())

            degrade_optimizer.zero_grad()
            degrade_loss.backward()
            degrade_optimizer.step()
        task_perf = check_accuracy(task_model, test_loader, TASK_LABEL_IDX, degrade_model)
        privacy_perf = check_accuracy(privacy_model, test_loader, PRIVACY_LABEL_IDX, degrade_model)
        print(epoch + 1, 'Degrade loss', np.round(np.mean(batch_loss), 3), 'Task acc', task_perf, 'Subj acc', privacy_perf)
    task_model.train(), privacy_model.train()


def train_task_classifier(model, train_loader, test_loader, degrade):
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    for epoch in range(EPOCHS):
        batch_loss = []
        for batch, item in enumerate(train_loader):
            inputs, targets = item[0].cuda(), item[TASK_LABEL_IDX].cuda()
            if degrade is not None:
                with torch.no_grad():
                    inputs = degrade(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        perf = check_accuracy(model, test_loader, TASK_LABEL_IDX, degrade)
        print(epoch + 1, 'Task loss', np.round(np.mean(batch_loss), 3), 'acc', perf)


def train_privacy_classifier(model, train_loader, test_loader, degrade):
    """ Outputs True if the model needs to be reset."""
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    epoch_loss = []
    for epoch in range(EPOCHS):
        batch_loss = []
        for batch, item in enumerate(train_loader):
            inputs, targets = item[0].cuda(), item[PRIVACY_LABEL_IDX].cuda()
            if degrade is not None:
                with torch.no_grad():
                    inputs = degrade(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        perf = check_accuracy(model, test_loader, PRIVACY_LABEL_IDX, degrade)
        print(epoch + 1, 'Subject loss', np.round(np.mean(batch_loss), 3), 'acc', perf)
        epoch_loss.append(np.mean(batch_loss))
        if epoch >= 5 and abs(epoch_loss[-5] - epoch_loss[-1]) < 0.001:
            print(abs(epoch_loss[-5] - epoch_loss[-1]), epoch_loss[-5:])
            return True
    return False


def check_accuracy(model, loader, task, degrade):
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for batch, item in enumerate(loader):
            inputs, targets = item[0].cuda(), item[task].cuda()
            if degrade is not None:
                inputs = degrade(inputs)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        perf = correct / total
    model.train()
    return np.round(perf*100, 2)


if __name__ == '__main__':
    main()
