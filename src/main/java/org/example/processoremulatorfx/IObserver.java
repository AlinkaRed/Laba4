package org.example.processoremulatorfx;

import org.example.processoremulatorfx.Models.CpuModel;
import org.example.processoremulatorfx.Models.ExecuterModel;
import org.example.processoremulatorfx.Models.ProgramModel;

public interface IObserver {
    void event(ProgramModel m);
    void event(CpuModel c);
    void event(ExecuterModel e);
}
