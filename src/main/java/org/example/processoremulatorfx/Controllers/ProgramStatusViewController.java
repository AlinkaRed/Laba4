package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import org.example.processoremulatorfx.BModels.BCpuModel;
import org.example.processoremulatorfx.BModels.BExecuterModel;
import org.example.processoremulatorfx.BModels.BProgramModel;
import org.example.processoremulatorfx.IObserver;
import org.example.processoremulatorfx.Models.CpuModel;
import org.example.processoremulatorfx.Models.ExecuterModel;
import org.example.processoremulatorfx.Models.ProgramModel;

public class ProgramStatusViewController implements IObserver {
  ProgramModel programModel = BProgramModel.build();
  CpuModel cpuModel = BCpuModel.build();
  ExecuterModel executerModel = BExecuterModel.build();

  @FXML
  void initialize() {
    programModel.addObserver(this);
    cpuModel.addObserver(this);
    executerModel.addObserver(this);
    proc.setText("none");
    step.setText("1");
    commands.setText("0");
    error.setText("0");
    memory.setText("not use");
  }

  @FXML
  Label proc;
  @FXML
  Label step;
  @FXML
  Label commands;
  @FXML
  Label error;
  @FXML
  Label memory;

  @Override
  public void event(ExecuterModel e) {
    proc.setText("none");
    if (e.getIter() > 0) proc.setText("Выполняется");
    if (e.getIsEnd()) proc.setText("Завершена");
    if (e.getIsDebug()) proc.setText(proc.getText() + "-О");
    step.setText(String.valueOf(e.getIter()+1));
    error.setText(String.valueOf(e.getErrors()));
  }

  @Override
  public void event(ProgramModel m) {
    commands.setText(String.valueOf(m.countCommads()));
  }

  @Override
  public void event(CpuModel c) {
    int bite = 0;
    for (int m : c.getMemory()) if (m != 0) bite+=4;
    var str = "";
    if (bite == 4) str = "a"; else str = "ов";
    memory.setText(String.valueOf(bite) + " байт" + str);
  }
}
