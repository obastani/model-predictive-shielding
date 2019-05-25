%% Setup

% cartpole parameters
g  = 9.8;
mc = 1.0;
mp = 0.1;
l  = 1.0;

% point to check
s = 0.01;

%% Nonlinear dynamics

% symbolic variables
syms z v t w u;
x = [ z ;
      v ;
      t ;
      w ];

% equations of motion
tmp = (u + l*(w^2)*sin(t)) / (mp + mc);
wdot = (g*sin(t) - tmp*cos(t)) / (l * (4.0/3.0 - mp*(cos(t)^2)/(mp + mc)));
vdot = tmp - l*wdot*cos(t) / (mp + mc);

% nonlinear dynamics
f = [ v    ;
      vdot ;
      w    ;
      wdot ];

%% Polynomial dynamics

g = subs(f, t, 0) ...
    + subs(diff(f, t), t, 0) * t ...
    + (1/2) * subs(diff(f, t, 2), t, 0) * t^2 ...
    + (1/6) * subs(diff(f, t, 3), t, 0) * t^3 ...
    + (1/24) * subs(diff(f, t, 4), t, 0) * t^4 ...
    + (1/120) * subs(diff(f, t, 5), t, 0) * t^5;

%% LQR

% origin
x0 = zeros([4, 1]);
u0 = 0.0;

% linear dynamics
A = double(subs(jacobian(f, x), [x; u], [x0; u0]));
B = double(subs(jacobian(f, u), [x; u], [x0; u0]));

% costs
Q = eye(4);
R = 0.1;
N = zeros([4, 1]);

% lqr
[K, S, E] = lqr(A, B, Q, R, N);

%% Regional stability

% candidate Lyapunov function
V = x.' * S * x;

% derivative
Vdot = jacobian(V, x) * subs(g, u, -K*x);

% sos program
syms epsilon;
prog = sosprogram(x, epsilon);

% lagrange multiplier
[prog, lambda] = sospolyvar(prog, monomials(x, 0:6));
[prog, mu] = sospolyvar(prog, monomials(x, 0:6));

% sos constraints (Vdot(x) < 0 for all x != 0 s.t. V(x) <= epsilon)
%
% Note: To avoid dependence on the time step dt, we use the differential
% version of the dynamics, LQR, and Lyapunov equation. This approximation
% is very good as long as dt is small.
%
% Note: It is easy to show that these constraints are equivalent to the
% original ones for LQR verification.
prog = sosineq(prog, - lambda*Vdot - (epsilon - V)*(t^2 + w^2)^2);
prog = sosineq(prog, - mu*(t^2 - 0.0225) - (epsilon - V)*(t^2 + w^2)^2);
prog = sosineq(prog, lambda);
prog = sosineq(prog, mu);

% minimize -epsilon
prog = sossetobj(prog, -epsilon);

% solve the program
prog = sossolve(prog);

% get the solution
epsilon = sosgetsol(prog, epsilon);

%% Results

x_cur = s * ones([4, 1]);
v_cur = double(subs(V, x, x_cur));

epsilon
v_cur

% Claim: Because v_cur <= epsilon, we have
%
%    { x | || x ||_infty <= s } \subseteq G_epsilon = { x | V(x) <= epsilon }
%
% Proof: Note that
%
%  - V has the form V(x) = || Lambda * x ||_2^2
%  - thus, V is invariant under reflection across any axis
%  - thus, V is convex
%
% Furthermore, v_cur <= epsilon, so
%
%    x_cur \in G_epsilon 
%
% By the above, G_epsilon contains the convex hull of x_cur and its
% reflections across any set of axes. QED
%


