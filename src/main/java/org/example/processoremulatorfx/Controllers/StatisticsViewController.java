package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import org.example.processoremulatorfx.BModels.BProgramModel;
import org.example.processoremulatorfx.IObserver;
import org.example.processoremulatorfx.Models.CpuModel;
import org.example.processoremulatorfx.Models.ExecuterModel;
import org.example.processoremulatorfx.Models.ProgramModel;

public class StatisticsViewController implements IObserver {
  ProgramModel programModel = BProgramModel.build();
  Font monoFont = Font.font("monospaced", 12);

  @FXML
  void initialize()
  {
    programModel.addObserver(this);
  }

  @FXML
  VBox Stat;

  @Override
  public void event(ProgramModel m) {
    Stat.getChildren().clear();
    Text st = new Text(m.MostPopularInstruction());
    st.setFont(monoFont);
    Stat.getChildren().add(new VBox(st));
  }

  @Override
  public void event(CpuModel c) {

  }

  @Override
  public void event(ExecuterModel e) {

  }
}
