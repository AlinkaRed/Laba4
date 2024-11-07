package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Text;
import javafx.scene.text.TextFlow;
import org.example.processoremulatorfx.*;
import org.example.processoremulatorfx.BModels.BExecuterModel;
import org.example.processoremulatorfx.BModels.BProgramModel;
import org.example.processoremulatorfx.Models.CpuModel;
import org.example.processoremulatorfx.Models.ExecuterModel;
import org.example.processoremulatorfx.Models.ProgramModel;

public class CommandViewController implements IObserver {
  ProgramModel programModel = BProgramModel.build();
  ExecuterModel executerModel = BExecuterModel.build();

  Command c;

  public void setCommand(Command c){
    this.c = c;

    Com.setText(c.getCommand());
    Arg1.setText(c.getArgs(0));
    Arg2.setText(c.getArgs(1));
  }

  @FXML
  Label Com;
  @FXML
  Label Arg1;
  @FXML
  Label Arg2;

  @FXML
  void initialize() { executerModel.addObserver(this); }

  @FXML
  void remove() { programModel.removeCommand(c); }

  @FXML
  void  down() throws Exception {
    try {
      programModel.switchCommandDown(c);
    }catch (Exception e){
      System.out.println("Warning: " + e.getMessage());
    }
  }

  @FXML
  void up() throws Exception {
    try {
      programModel.switchCommandUp(c);
    }catch (Exception e){
      System.out.println("Warning: " + e.getMessage());
    }
  }

  @Override
  public void event(ProgramModel m) {}

  @Override
  public void event(CpuModel c) {}

  @Override
  public void event(ExecuterModel e) {
    if (c != null) {
      if (e.getIter() == programModel.getNumCommand(c))
        setColorCom(Color.RED);
      else
        setColorCom(Color.BLACK);
    }
  }

  private void setColorCom(Color color){
    Com.setTextFill(color);
  }
}
