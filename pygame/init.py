import pygame

WIDTH = 360
HEIGHT = 480
FPS = 30

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.init()
pygame.mixer.init() # for sound
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My game")
clock = pygame.time.Clock()

running = True
while running:
    clock.tick(FPS)
    #Process input
    for event in pygame.event.get():
    # check for close window
        if event.type == pygame.QUIT:
            running = False
    #Update
    #Render (draw)
    screen.fill(RED)
    pygame.display.flip()

pygame.quit()
