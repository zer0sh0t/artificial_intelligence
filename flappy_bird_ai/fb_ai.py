from fb_engine import *
import neat
import os

global scores
global high_score
global max_gen
scores = []
max_gen = 200


def fitness_fn(genomes, config):
    global WIN, gen
    win = WIN
    gen += 1
    nets = []
    birds = []
    g = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        if score > 200 or gen == max_gen:
            clock.tick(30)
        else:
            clock.tick(50000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].TOP_PIPE.get_width():
                pipe_ind = 1

        for i, bird in enumerate(birds):
            g[i].fitness += 0.1
            bird.move()

            output = nets[i].activate((bird.y, abs(bird.y - pipes[pipe_ind].top),
                                       abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()

        base.move()

        r = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()

            for i, bird in enumerate(birds):
                if pipe.collide(bird, win):
                    g[i].fitness -= 1
                    nets.pop(i)
                    birds.pop(i)
                    g.pop(i)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.TOP_PIPE.get_width() < 0:
                r.append(pipe)

        if add_pipe:
            score += 1
            for genome in g:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for pipe in r:
            pipes.remove(pipe)

        for i, bird in enumerate(birds):
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(i)
                birds.pop(i)
                g.pop(i)

        scores.append(score)
        high_score = max(scores)

        for s in scores:
            if s != high_score:
                scores.pop(scores.index(s))

        draw_window(win, birds, pipes, base, score, high_score, gen, pipe_ind)

    if gen == max_gen:
        print("Best Score: " + str(high_score))
        print("")

    print("Score: " + str(score))
    print("")


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    best_genome = p.run(fitness_fn, max_gen)
    print("\nBest Genome:\n{!s}".format(best_genome))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
