import pygame
import os
import random


SCREEN_WIDTH = 750
SCREEN_HEIGHT = 650

FPS = 60

class FlameParticle:
    alpha_layer_qty = 2
    alpha_glow_difference_constant = 3
    max_radius = 4

    def __init__(self, screen, x=SCREEN_WIDTH // 2, y=SCREEN_HEIGHT // 2, r=max_radius):
        self.x = x
        self.y = y
        self.r = r
        self.original_r = r
        self.alpha_layers = FlameParticle.alpha_layer_qty
        self.alpha_glow = FlameParticle.alpha_glow_difference_constant
        max_surf_size = 2 * self.r * self.alpha_layers * self.alpha_layers * self.alpha_glow
        self.surface = pygame.Surface((max_surf_size, max_surf_size), pygame.SRCALPHA)
        self.burn_rate = 0.1 * random.randint(1, 4)
        self.screen = screen

    def update(self):
        self.y -= 7 - self.r
        self.x += random.randint(-self.r, self.r)
        self.original_r -= self.burn_rate
        self.r = int(self.original_r)
        if self.r <= 0:
            self.r = 1

    def draw(self):
        max_surf_size = 2 * self.r * self.alpha_layers * self.alpha_layers * self.alpha_glow
        self.surface = pygame.Surface((max_surf_size, max_surf_size), pygame.SRCALPHA)
        for i in range(self.alpha_layers, -1, -1):
            alpha = 255 - i * (255 // self.alpha_layers - 5)
            if alpha <= 0:
                alpha = 0
            radius = self.r * i * i * self.alpha_glow
            if self.r == 4 or self.r == 3:
                r, g, b = (232, 100, 65)
            elif self.r == 2:
                r, g, b = (232, 197, 109)
            else:
                r, g, b = (92, 50, 50)
            r, g, b = (b, g, r)
            color = (r, g, b, alpha)
            pygame.draw.circle(self.surface, color, (self.surface.get_width() // 2, self.surface.get_height() // 2), radius)
        self.screen.blit(self.surface, self.surface.get_rect(center=(self.x, self.y)))


class Flame:
    def __init__(
        self,
        screen,
        x=SCREEN_WIDTH // 2,
        y=SCREEN_HEIGHT // 2,
        flame_intensity=3,
        intensity_ratio=25,
        max_radius=4,
    ):
        self.x = x
        self.y = y
        self.flame_intensity = flame_intensity
        self.intensity_ratio = intensity_ratio
        self.max_radius = max_radius
        self.flame_particles = []
        self.screen = screen
        for i in range(self.flame_intensity * self.intensity_ratio):
            self.flame_particles.append(
                FlameParticle(
                    screen,
                    self.x + random.randint(-5, 5),
                    self.y,
                    random.randint(1, FlameParticle.max_radius),
                )
            )

    def draw_flame(self):
        for f in self.flame_particles:
            if f.original_r <= 0:
                self.flame_particles.remove(f)
                self.flame_particles.append(
                    FlameParticle(
                        self.screen,
                        self.x + random.randint(-5, 5),
                        self.y,
                        random.randint(1, self.max_radius),
                    )
                )
                del f
                continue
            f.update()
            f.draw()
