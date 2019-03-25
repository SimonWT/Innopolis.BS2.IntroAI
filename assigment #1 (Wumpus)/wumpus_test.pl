%Set walls with direction
% wall(_,1, south).
% wall(_,5, north).
% wall(1,_, west).
% wall(5,_, east).

%Set walls withput direction
wall(X, Y) :-
    X > 5; Y > 5; X < 1; Y < 1.

%Set Gold, Wumpus, PITs
glint(3,3).
wumpus(1,2).
pit(2,2).

%Set defaults, if we don't have objects on the map
pit(X, Y):-
    false.

%Markers
breeze(X, Y) :-
    NUY is Y + 1, NDY is Y - 1, NRX is X + 1, NLX is X - 1, 
    (pit(NRX, Y); pit(NLX, Y); pit(X, NUY); pit(X, NDY)).

stench(X, Y ):-
    NUY is Y + 1, NDY is Y - 1, NRX is X + 1, NLX is X - 1, 
    (wumpus(NRX, Y); wumpus(NLX, Y); wumpus(X, NUY); wumpus(X, NDY)).

%Pick Up the gold from the [X,Y] block
pickup(X, Y) :-
    glint(X, Y).

%Check the Arrow
hasArrow(Arrow):-
    Arrow > 0.

%set begin agent position (1,1)
agent(1,1).

%Safe position X,Y or not
safe(X, Y) :-
    \+wumpus(X,Y),
    \+pit(X,Y).
    %\+breeze(X,Y),
    %\+stench(X,Y).

%Actions that, agent can perform
%action (CurrentX, CurrentY, NewX, NewY, Path, add path to the List)

action(X, Y, X, NewY, List, [up|List]) :- 
    Y < 3,
    NewY is Y + 1,
    safe(X, NewY).

action(X, Y, X, NewY, List, [down|List]) :- 
    Y > 1,
    NewY is Y - 1,
    safe(X, NewY).

action(X, Y, NewX, Y, List, [right|List]) :- 
    X < 3,
    NewX is X + 1,
    safe(NewX, Y).

action(X, Y, NewX, Y, List, [left|List]) :- 
    X > 1,
    NewX is X - 1,
    safe(NewX, Y).


%Recursive query that find paths and write the steps into the list
getPathRec(X, Y, VisitedList, CurrentList, List) :-
    \+member((X, Y), VisitedList),

    action(X, Y, NewX, NewY, CurrentList, NewList),

    getPathRec(NewX, NewY, [(X, Y) | VisitedList], NewList, List).

  % Stop find new path if there is gold, and reverse path to make it readable.
getPathRec(X, Y, _ , List, NewList) :-
    glint(X, Y),
    reverse(List, NewList).

%main query to find the path
findPath(Path):-
    agent(X, Y),
    getPathRec(X, Y, [], [], Path).

