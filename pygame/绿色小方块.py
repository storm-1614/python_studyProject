"""
让屏幕中间出现绿色小方块在移动
"""

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

class Player (pygame.sprite.Sprite):
    def __init__(self):
        # Runs the built-in Sprite classes initializer.
        pygame.sprite.Sprite.__init__(self)
        # Define the image property
        # Creating a simple 50x50 squre and filling it with the color GREEN
        self.image = pygame.Surface((50, 50))
        self.image.fill(GREEN)
        # 创建一个矩形
        # self.rect 实际是 Rect 这个类，包含很多东西
        self.rect = self.image.get_rect()
        self.rect.center = (int(WIDTH / 2), int(HEIGHT / 2))

    def update(self):
        self.rect.x += 5
        if self.rect.right > WIDTH:
            self.rect.left = 0

all_sprites = pygame.sprite.Group()
player = Player()
all_sprites.add(player)


running = True
while running:
    clock.tick(FPS)
    #Process input
    for event in pygame.event.get():
    # check for close window
        if event.type == pygame.QUIT:
            running = False
    #Update
    all_sprites.update()
    #Render (draw)
    screen.fill(BLACK)
    all_sprites.draw(screen)
    pygame.display.flip()

pygame.quit()
