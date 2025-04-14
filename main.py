from time import time

from app import App


if __name__ == '__main__':
    start_time = time()

    app = App(
        size=(96, 50), resolution=40, radius=0.05, max_velocity=5
    )
    app.generate(100)
    app.save(file_name='output.png')

    print(f'Time: {time() - start_time} seconds')
