gold(3, 3).
agent(1, 1).

getAllPaths(MS, L) :-
    agent(X, Y),
    retractall(worldSize(_)),
    assertz(worldSize(MS)),
    getAllPathsRec(X, Y, [], [], L).


% Positions, Visited list, and Path list
getAllPathsRec(X, Y, _V, L, NL) :-
    gold(X, Y),
    reverse(L, NL).


% Positions, Visited list, and Path list
getAllPathsRec(X, Y, V, CL, L) :-
    \+member((X, Y), V),

    % useless 
    % append(V, [(X, Y)], VP),

    move(X, Y, CL, NX, NY, NL),

    % No need to use append to build the list of visited nodes
    getAllPathsRec(NX, NY, [(X,Y) | V], NL, L).

% Left
move(X, Y, L, NX, Y, [l|L]) :-
    X > 1 ,NX is X - 1.

% Right
move(X, Y, L, NX, Y, [r|L]) :-
    worldSize(MS), X < MS,NX is X + 1.

% Up
move(X, Y, L, X, NY, [u|L]) :-
    worldSize(MS), Y < MS, NY is Y + 1.

% Down
move(X, Y, L, X, NY, [d|L]) :-
    Y > 1, NY is Y - 1.