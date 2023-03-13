import motorModule
import curses

screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)

motor= motorModule.Motor(17,27,10,9)

try:
    while True:
        cont = screen.getch()
        if cont == ord('q'):
            motor.stop(0.1)
        if cont==curses.KEY_UP:
            motor.drive(20,20)
            motor.stop(0.01)
        if cont==curses.KEY_LEFT:
            # motor.stop(0.1)
            motor.drive(0,20)
            motor.stop(0.1)
        if cont==curses.KEY_RIGHT:
            # motor.stop(0.1)
            motor.drive(20,0,)
            motor.stop(0.1)
        elif cont==curses.KEY_DOWN:
            motor.drive(-20,-20)
            motor.stop(0.1)
        # elif cont==:
        #     motor.move(0.2,0.8,1)
    motor.stop(0.1)
finally:
    curses.nocbreak();screen.keypad(0);curses.echo()
    curses.endwin()