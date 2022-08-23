from TrainingHandler import TrainingHandler

if __name__ == '__main__':
    dev_note = f'8t 128x2 reproduce'
    h = TrainingHandler(dev_note=dev_note)
    h.run()

