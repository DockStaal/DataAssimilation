function wave1d_plotseries(t,series_data,s,obs_data);
%function wave1d_plotseries(t,series_data,s,obs_data);
%plot timeseries from model and observations
loc_names=s.loc_names;
nseries=length(loc_names);
for i=1:nseries,
    disp(i)
    figure(i+1);clf;
    plot(t,series_data(i,:),'b-');
    hold on;
    ntimes=min(length(t),size(obs_data,2));
    plot(t(1:ntimes),obs_data(i,2:(ntimes+1)),'k-'); %observations also contain initial time
    title(loc_names{i});
    xlabel('time');
    hold off;
    datetick('x');
    %print(replace(sprintf('%s.png',loc_names{i}),' ','_'),'-dpng');
end;
