function nmopso_save_data(filename, data)
    if isstruct(data)
        dt_sv = data;
        save(filename, 'dt_sv');
    else
        gen_hv = data;
        save(filename, 'gen_hv');
    end
end
