package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import org.example.processoremulatorfx.BModels.BCpuModel;
import org.example.processoremulatorfx.IObserver;
import org.example.processoremulatorfx.Models.CpuModel;
import org.example.processoremulatorfx.Models.ExecuterModel;
import org.example.processoremulatorfx.Models.ProgramModel;

public class RegistersViewController implements IObserver {
  CpuModel cpuModel = BCpuModel.build();

  @FXML
  void initialize()
  {
    cpuModel.addObserver(this);
  }

  @FXML
  Label regA;
  @FXML
  Label regB;
  @FXML
  Label regC;
  @FXML
  Label regD;

  @Override
  public void event(CpuModel c) {
    regA.setText(String.valueOf(c.getRegisters()[0]));
    regB.setText(String.valueOf(c.getRegisters()[1]));
    regC.setText(String.valueOf(c.getRegisters()[2]));
    regD.setText(String.valueOf(c.getRegisters()[3]));
  }

  @Override
  public void event(ExecuterModel e) {

  }

  @Override
  public void event(ProgramModel m) {

  }
}
