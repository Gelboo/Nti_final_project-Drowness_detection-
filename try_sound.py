import pygame

import time

pygame.init()

pygame.mixer.music.load("truckAlarm.wav")

pygame.mixer.music.play(-1)

#pygame.mixer.music.queue('truckAlarm.wav')

time.sleep(10)
pygame.mixer.music.stop()
