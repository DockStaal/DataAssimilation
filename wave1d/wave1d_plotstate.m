function wawe1d_plotstate(fig,x,i,s);
%function wawe1d_plotstate(fig,x,i,s);
%plot all waterlevels and velocities at one time
figure(fig);
clf;
xh=s.x_h;
subplot(2,1,1);
plot(xh,x(1:2:end))
ylabel('h')
xu=s.x_u;
subplot(2,1,2);
plot(xu,x(2:2:end));
ylabel('u');
print(sprintf('fig_map_%3.3d.png',i),'-dpng');
pause(0.2);
